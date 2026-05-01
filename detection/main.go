package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os/signal"
	"syscall"
	"time"

	"fraud-engine/detection/features"
	"fraud-engine/detection/fingerprint"
	"fraud-engine/detection/graph"
	"fraud-engine/detection/inference"
	"fraud-engine/detection/metrics"
	"fraud-engine/detection/publisher"
	"fraud-engine/detection/threshold"
	"fraud-engine/shared"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
	"github.com/twmb/franz-go/pkg/kgo"
)

const brokerAddr = "localhost:9092"

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		http.Handle("/metrics", promhttp.Handler())
		log.Println("Metrics server started on :2112")
		if err := http.ListenAndServe(":2112", nil); err != nil {
			log.Printf("metrics server error: %v", err)
		}
	}()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		PoolSize: 20,
	})
	defer rdb.Close()

	agg := features.New(rdb)
	fpStore := fingerprint.New(rdb)
	threshCalc := threshold.New(rdb)
	ringDetector := graph.New(rdb)

	scorer, err := inference.New()
	if err != nil {
		log.Fatalf("failed to create scorer: %v", err)
	}

	pub, err := publisher.New(brokerAddr)
	if err != nil {
		log.Fatalf("failed to create publisher: %v", err)
	}
	defer pub.Close()

	consumer, err := kgo.NewClient(
		kgo.SeedBrokers(brokerAddr),
		kgo.ConsumerGroup("detection-service"),
		kgo.ConsumeTopics("transactions"),
		kgo.FetchMinBytes(1),
	)
	if err != nil {
		log.Fatalf("failed to create consumer: %v", err)
	}
	defer consumer.Close()

	log.Println("Detection service started — listening for transactions...")

	for {
		fetches := consumer.PollFetches(ctx)

		if fetches.IsClientClosed() {
			return
		}

		if ctx.Err() != nil {
			return
		}

		if err := fetches.Err(); err != nil {
			if ctx.Err() != nil {
				return
			}
			metrics.ProcessingErrors.Inc()
			log.Printf("fetch error: %v", err)
			continue
		}

		fetches.EachRecord(func(r *kgo.Record) {
			start := time.Now()

			var tx shared.Transaction
			if err := json.Unmarshal(r.Value, &tx); err != nil {
				metrics.ProcessingErrors.Inc()
				return
			}

			// Sliding window features
			feats, err := agg.Aggregate(ctx, tx)
			if err != nil {
				metrics.ProcessingErrors.Inc()
				return
			}

			// Behavioral anomaly
			anomaly, _ := fpStore.ScoreAnomaly(ctx, tx)
			fpStore.UpdateProfile(ctx, tx)

			// Adaptive threshold
			threshCalc.UpdateStats(ctx, tx.CardID, tx.AmountUSD)
			adaptiveThreshold := threshCalc.GetThreshold(ctx, tx.CardID, tx.AmountUSD)

			// Graph fraud ring detection
			ringDetector.RecordTransaction(ctx, tx.CardID, tx.MerchantID, tx.AmountUSD)
			ringDetector.TrackCardMerchant(ctx, tx.CardID, tx.MerchantID)
			ringSignal, _ := ringDetector.DetectRing(ctx, tx.CardID, tx.MerchantID)

			// ML model score
			mlScore, _ := scorer.Score(feats)

			// Combined score:
			// 60% ML model
			// 25% behavioral fingerprint
			// 15% graph ring signal
			combinedScore := float32(
				float64(mlScore)*0.60 +
					float64(anomaly.OverallAnomaly)*0.25 +
					float64(ringSignal.RiskScore)*0.15,
			)

			isFraud := combinedScore >= adaptiveThreshold

			metrics.TransactionsTotal.Inc()
			metrics.FraudScoreHistogram.Observe(float64(combinedScore))

			elapsed := time.Since(start)
			metrics.DetectionLatency.Observe(float64(elapsed.Milliseconds()))

			if isFraud {
				metrics.FraudAlertsTotal.Inc()
				alert := shared.FraudAlert{
					TransactionID: tx.ID,
					CardID:        tx.CardID,
					FraudScore:    combinedScore,
					DetectedAt:    time.Now().UTC(),
					Features:      feats,
				}
				if err := pub.Publish(ctx, alert); err != nil {
					metrics.ProcessingErrors.Inc()
				}
				if ringSignal.RiskScore > 0 {
					log.Printf("🕸️  RING DETECTED | merchant: %s | cards in 30min: %d | ring score: %.2f",
						tx.MerchantID, ringSignal.CardCount, ringSignal.RiskScore)
				}
			}

			if elapsed > 40*time.Millisecond {
				log.Printf("WARN slow: %s took %v", tx.ID, elapsed)
			} else {
				log.Printf("processed: %s | ml: %.2f | anomaly: %.2f | ring: %.2f | combined: %.2f | threshold: %.2f | fraud: %v | latency: %v",
					tx.ID, mlScore, anomaly.OverallAnomaly, ringSignal.RiskScore,
					combinedScore, adaptiveThreshold, isFraud, elapsed)
			}
		})
	}
}
