package threshold

import (
	"context"
	"encoding/json"
	"fmt"
	"math"

	"github.com/redis/go-redis/v9"
)

// CardThreshold stores adaptive threshold config per card
type CardThreshold struct {
	CardID         string  `json:"card_id"`
	BaseThreshold  float32 `json:"base_threshold"`
	FalsePositives int     `json:"false_positives"`
	TruePositives  int     `json:"true_positives"`
	TxnCount       int64   `json:"txn_count"`
	AvgSpend       float64 `json:"avg_spend"`
	SpendStdDev    float64 `json:"spend_std_dev"`
	SumSquares     float64 `json:"sum_squares"`
}

type Calculator struct {
	rdb             *redis.Client
	globalThreshold float32
}

func New(rdb *redis.Client) *Calculator {
	return &Calculator{
		rdb:             rdb,
		globalThreshold: 0.5,
	}
}

func thresholdKey(cardID string) string {
	return fmt.Sprintf("threshold:%s", cardID)
}

// GetThreshold returns the adaptive threshold for a card
func (c *Calculator) GetThreshold(ctx context.Context, cardID string, amount float64) float32 {
	data, err := c.rdb.Get(ctx, thresholdKey(cardID)).Bytes()
	if err != nil {
		return c.globalThreshold
	}

	var ct CardThreshold
	if err := json.Unmarshal(data, &ct); err != nil {
		return c.globalThreshold
	}

	// Not enough history — use global threshold
	if ct.TxnCount < 10 {
		return c.globalThreshold
	}

	threshold := float64(c.globalThreshold)

	// If card has high average spend — raise threshold
	// (they legitimately make large purchases)
	if ct.AvgSpend > 500 {
		threshold += 0.1
	} else if ct.AvgSpend > 200 {
		threshold += 0.05
	}

	// If card has consistent spend (low std dev) — lower threshold
	// (unusual amounts are more suspicious)
	if ct.SpendStdDev < 50 && ct.AvgSpend < 200 {
		threshold -= 0.1
	}

	// If this amount is way above their normal — lower threshold
	// (be more aggressive about flagging)
	if ct.AvgSpend > 0 {
		ratio := amount / ct.AvgSpend
		if ratio > 5 {
			threshold -= 0.15
		} else if ratio > 3 {
			threshold -= 0.08
		}
	}

	// Clamp between 0.3 and 0.8
	threshold = math.Max(0.3, math.Min(0.8, threshold))

	return float32(threshold)
}

// UpdateStats updates the card's spending statistics
func (c *Calculator) UpdateStats(ctx context.Context, cardID string, amount float64) error {
	data, err := c.rdb.Get(ctx, thresholdKey(cardID)).Bytes()

	var ct CardThreshold
	if err == nil {
		json.Unmarshal(data, &ct)
	} else {
		ct = CardThreshold{
			CardID:        cardID,
			BaseThreshold: c.globalThreshold,
		}
	}

	// Update running mean and std dev using Welford's algorithm
	ct.TxnCount++
	delta := amount - ct.AvgSpend
	ct.AvgSpend += delta / float64(ct.TxnCount)
	delta2 := amount - ct.AvgSpend
	ct.SumSquares += delta * delta2

	if ct.TxnCount > 1 {
		ct.SpendStdDev = math.Sqrt(ct.SumSquares / float64(ct.TxnCount-1))
	}

	updated, err := json.Marshal(ct)
	if err != nil {
		return err
	}

	return c.rdb.Set(ctx, thresholdKey(cardID), updated, 0).Err()
}
