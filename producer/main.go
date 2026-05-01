package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"time"

	"fraud-engine/shared"

	"github.com/google/uuid"
	"github.com/twmb/franz-go/pkg/kgo"
)

const (
	brokerAddr  = "localhost:9092"
	topic       = "transactions"
	ratePerSec  = 500
	workerCount = 16
)

var countries = []string{"US", "GB", "DE", "IN", "BR", "CN", "RU"}
var merchants = []string{"merch-001", "merch-002", "merch-003", "merch-004", "merch-005"}

func syntheticTransaction() shared.Transaction {
	isFraudulent := rand.Float64() < 0.05
	amount := 10.0 + rand.Float64()*200
	country := "US"

	if isFraudulent {
		amount = 500 + rand.Float64()*2000
		country = countries[1+rand.Intn(len(countries)-1)]
	}

	return shared.Transaction{
		ID:          uuid.NewString(),
		CardID:      fmt.Sprintf("card-%d", rand.Intn(100)),
		MerchantID:  merchants[rand.Intn(len(merchants))],
		AmountUSD:   amount,
		MCC:         rand.Intn(9999),
		CountryCode: country,
		Timestamp:   time.Now().UTC(),
	}
}

func main() {
	client, err := kgo.NewClient(
		kgo.SeedBrokers(brokerAddr),
		kgo.RecordPartitioner(kgo.StickyKeyPartitioner(nil)),
		kgo.ProducerLinger(5*time.Millisecond),
	)
	if err != nil {
		log.Fatalf("failed to create kafka client: %v", err)
	}
	defer client.Close()

	log.Printf("Producer started — sending %d transactions/sec", ratePerSec)

	ticker := time.NewTicker(time.Second / ratePerSec)
	defer ticker.Stop()

	jobs := make(chan struct{}, workerCount*10)

	for i := 0; i < workerCount; i++ {
		go func() {
			for range jobs {
				tx := syntheticTransaction()
				payload, err := json.Marshal(tx)
				if err != nil {
					log.Printf("marshal error: %v", err)
					continue
				}

				record := &kgo.Record{
					Topic: topic,
					Key:   []byte(tx.CardID),
					Value: payload,
				}

				if err := client.ProduceSync(context.Background(), record).FirstErr(); err != nil {
					log.Printf("produce error: %v", err)
				} else {
					log.Printf("sent: %s | card: %s | $%.2f | %s",
						tx.ID, tx.CardID, tx.AmountUSD, tx.CountryCode)
				}
			}
		}()
	}

	for range ticker.C {
		jobs <- struct{}{}
	}
}
