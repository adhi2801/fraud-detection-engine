package shared

import "time"

type Transaction struct {
	ID          string    `json:"id"`
	CardID      string    `json:"card_id"`
	MerchantID  string    `json:"merchant_id"`
	AmountUSD   float64   `json:"amount_usd"`
	MCC         int       `json:"mcc"`
	CountryCode string    `json:"country_code"`
	Timestamp   time.Time `json:"timestamp"`
}

type FraudAlert struct {
	TransactionID string    `json:"transaction_id"`
	CardID        string    `json:"card_id"`
	FraudScore    float32   `json:"fraud_score"`
	DetectedAt    time.Time `json:"detected_at"`
	Features      Features  `json:"features"`
}

type Features struct {
	AmountUSD       float32 `json:"amount_usd"`
	TxnCount1h      float32 `json:"txn_count_1h"`
	TotalSpend1h    float32 `json:"total_spend_1h"`
	DistinctMerch1h float32 `json:"distinct_merch_1h"`
	IsInternational float32 `json:"is_international"`
}
