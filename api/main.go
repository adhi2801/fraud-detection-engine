package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"fraud-engine/detection/features"
	"fraud-engine/detection/fingerprint"
	"fraud-engine/detection/graph"
	"fraud-engine/detection/inference"
	"fraud-engine/detection/threshold"
	"fraud-engine/shared"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

type EvaluateRequest struct {
	CardID      string  `json:"card_id"`
	AmountUSD   float64 `json:"amount_usd"`
	MerchantID  string  `json:"merchant_id"`
	CountryCode string  `json:"country_code"`
}

type EvaluateResponse struct {
	TransactionID string   `json:"transaction_id"`
	CardID        string   `json:"card_id"`
	FraudScore    float32  `json:"fraud_score"`
	IsFraud       bool     `json:"is_fraud"`
	Threshold     float32  `json:"threshold"`
	Reasons       []string `json:"reasons"`
	LatencyMS     float64  `json:"latency_ms"`
}

type Server struct {
	agg          *features.Aggregator
	fpStore      *fingerprint.Store
	threshCalc   *threshold.Calculator
	ringDetector *graph.Detector
	scorer       *inference.Scorer
}

func main() {
	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		PoolSize: 10,
	})

	scorer, err := inference.New()
	if err != nil {
		log.Fatalf("failed to create scorer: %v", err)
	}

	srv := &Server{
		agg:          features.New(rdb),
		fpStore:      fingerprint.New(rdb),
		threshCalc:   threshold.New(rdb),
		ringDetector: graph.New(rdb),
		scorer:       scorer,
	}

	http.HandleFunc("/evaluate", srv.handleEvaluate)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	})

	log.Println("API server started on :8090")
	if err := http.ListenAndServe(":8090", nil); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

func (s *Server) handleEvaluate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()
	ctx := context.Background()

	var req EvaluateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}

	// Build transaction
	tx := shared.Transaction{
		ID:          uuid.NewString(),
		CardID:      req.CardID,
		MerchantID:  req.MerchantID,
		AmountUSD:   req.AmountUSD,
		CountryCode: req.CountryCode,
		Timestamp:   time.Now().UTC(),
	}

	// Get features
	feats, err := s.agg.Aggregate(ctx, tx)
	if err != nil {
		http.Error(w, "feature aggregation failed", http.StatusInternalServerError)
		return
	}

	// Get behavioral profile for explainability
	profile, _ := s.fpStore.GetProfile(ctx, req.CardID)

	// Get anomaly score
	anomaly, _ := s.fpStore.ScoreAnomaly(ctx, tx)

	// Get ring signal
	s.ringDetector.RecordTransaction(ctx, req.CardID, req.MerchantID, req.AmountUSD)
	s.ringDetector.TrackCardMerchant(ctx, req.CardID, req.MerchantID)
	ringSignal, _ := s.ringDetector.DetectRing(ctx, req.CardID, req.MerchantID)

	// Get adaptive threshold
	s.threshCalc.UpdateStats(ctx, req.CardID, req.AmountUSD)
	adaptiveThreshold := s.threshCalc.GetThreshold(ctx, req.CardID, req.AmountUSD)

	// ML score
	mlScore, _ := s.scorer.Score(feats)

	// Combined score
	combinedScore := float32(
		float64(mlScore)*0.60 +
			float64(anomaly.OverallAnomaly)*0.25 +
			float64(ringSignal.RiskScore)*0.15,
	)

	isFraud := combinedScore >= adaptiveThreshold

	// Build explainability reasons
	reasons := buildReasons(req, profile, anomaly, ringSignal, feats)

	elapsed := time.Since(start)

	resp := EvaluateResponse{
		TransactionID: tx.ID,
		CardID:        req.CardID,
		FraudScore:    combinedScore,
		IsFraud:       isFraud,
		Threshold:     adaptiveThreshold,
		Reasons:       reasons,
		LatencyMS:     float64(elapsed.Microseconds()) / 1000.0,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func buildReasons(
	req EvaluateRequest,
	profile *fingerprint.CardProfile,
	anomaly fingerprint.AnomalyScore,
	ring graph.RingSignal,
	feats shared.Features,
) []string {
	var reasons []string

	// Amount reasons
	if profile != nil && profile.AvgSpend > 0 {
		ratio := req.AmountUSD / profile.AvgSpend
		if ratio > 2 {
			reasons = append(reasons, fmt.Sprintf(
				"Amount $%.0f is %.1fx above card average of $%.0f",
				req.AmountUSD, ratio, profile.AvgSpend,
			))
		}
	}

	// Country reasons
	if anomaly.CountryAnomaly > 0 && profile != nil {
		reasons = append(reasons, fmt.Sprintf(
			"%s is a new country for this card (known: %v)",
			req.CountryCode, profile.Countries,
		))
	}

	// Ring reasons
	if ring.RiskScore > 0 {
		reasons = append(reasons, fmt.Sprintf(
			"%d cards hit merchant %s in the last 30 minutes",
			ring.CardCount, ring.MerchantID,
		))
	}

	// Velocity reasons
	if feats.TxnCount1h > 10 {
		reasons = append(reasons, fmt.Sprintf(
			"High transaction velocity: %.0f transactions in last hour",
			feats.TxnCount1h,
		))
	}

	// Spend reasons
	if feats.TotalSpend1h > 1000 {
		reasons = append(reasons, fmt.Sprintf(
			"High hourly spend: $%.0f in last hour",
			feats.TotalSpend1h,
		))
	}

	if len(reasons) == 0 {
		reasons = append(reasons, "Statistical anomaly detected by ML model")
	}

	return reasons
}
