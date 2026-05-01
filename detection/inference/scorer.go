package inference

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"fraud-engine/shared"
)

const (
	fraudThreshold = float32(0.5)
	inferenceURL   = "http://localhost:8888/score"
)

type Scorer struct {
	client *http.Client
}

type scoreRequest struct {
	AmountUSD       float32 `json:"amount_usd"`
	TxnCount1h      float32 `json:"txn_count_1h"`
	TotalSpend1h    float32 `json:"total_spend_1h"`
	DistinctMerch1h float32 `json:"distinct_merch_1h"`
	IsInternational float32 `json:"is_international"`
}

type scoreResponse struct {
	FraudScore float32 `json:"fraud_score"`
	IsFraud    bool    `json:"is_fraud"`
}

func New() (*Scorer, error) {
	// Health check client with longer timeout
	healthClient := &http.Client{
		Timeout: 5 * time.Second,
	}

	// Verify inference server is running
	resp, err := healthClient.Get("http://localhost:8888/health")
	if err != nil {
		return nil, fmt.Errorf("inference server not running: %w", err)
	}
	defer resp.Body.Close()

	// Scoring client with tight timeout for low latency
	client := &http.Client{
		Timeout: 30 * time.Millisecond,
	}

	return &Scorer{client: client}, nil
}

func (s *Scorer) Score(f shared.Features) (float32, bool) {
	req := scoreRequest{
		AmountUSD:       f.AmountUSD,
		TxnCount1h:      f.TxnCount1h,
		TotalSpend1h:    f.TotalSpend1h,
		DistinctMerch1h: f.DistinctMerch1h,
		IsInternational: f.IsInternational,
	}

	payload, err := json.Marshal(req)
	if err != nil {
		return s.ruleBasedScore(f)
	}

	resp, err := s.client.Post(inferenceURL, "application/json", bytes.NewBuffer(payload))
	if err != nil {
		// Fall back to rules if ML server is unreachable
		return s.ruleBasedScore(f)
	}
	defer resp.Body.Close()

	var result scoreResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return s.ruleBasedScore(f)
	}

	return result.FraudScore, result.IsFraud
}

// Fallback rule-based scorer if ML server fails
func (s *Scorer) ruleBasedScore(f shared.Features) (float32, bool) {
	score := float32(0)
	if f.AmountUSD > 500 {
		score += 0.4
	} else if f.AmountUSD > 200 {
		score += 0.2
	}
	if f.IsInternational == 1 {
		score += 0.3
	}
	if f.TxnCount1h > 10 {
		score += 0.2
	}
	if f.TotalSpend1h > 1000 {
		score += 0.2
	}
	if score > 1.0 {
		score = 1.0
	}
	return score, score >= fraudThreshold
}
