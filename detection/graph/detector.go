package graph

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
)

// RingSignal represents a detected fraud ring signal
type RingSignal struct {
	MerchantID    string
	CardCount     int
	TimeWindowMin int
	RiskScore     float32
}

type Detector struct {
	rdb *redis.Client
}

func New(rdb *redis.Client) *Detector {
	return &Detector{rdb: rdb}
}

// RecordTransaction adds card-merchant edge to the graph
func (d *Detector) RecordTransaction(ctx context.Context, cardID, merchantID string, amount float64) error {
	now := time.Now().UnixMilli()
	windowKey := fmt.Sprintf("merchant:%s:cards", merchantID)

	// Add this card to the merchant's recent card set
	// Score = timestamp so we can find cards that transacted recently
	pipe := d.rdb.Pipeline()
	pipe.ZAdd(ctx, windowKey, redis.Z{
		Score:  float64(now),
		Member: fmt.Sprintf("%s:%.2f", cardID, amount),
	})
	// Keep only last 30 minutes of data
	pipe.ZRemRangeByScore(ctx, windowKey, "0",
		strconv.FormatInt(now-30*60*1000, 10))
	pipe.Expire(ctx, windowKey, 31*time.Minute)
	_, err := pipe.Exec(ctx)
	return err
}

// DetectRing checks if current merchant has suspicious card clustering
func (d *Detector) DetectRing(ctx context.Context, cardID, merchantID string) (RingSignal, error) {
	windowKey := fmt.Sprintf("merchant:%s:cards", merchantID)

	// Get all cards that used this merchant in last 30 minutes
	members, err := d.rdb.ZRange(ctx, windowKey, 0, -1).Result()
	if err != nil {
		return RingSignal{}, err
	}

	// Count distinct cards
	distinctCards := map[string]struct{}{}
	for _, m := range members {
		// member format is "cardID:amount"
		for i, c := range m {
			if c == ':' {
				distinctCards[m[:i]] = struct{}{}
				break
			}
		}
	}

	cardCount := len(distinctCards)

	signal := RingSignal{
		MerchantID:    merchantID,
		CardCount:     cardCount,
		TimeWindowMin: 30,
	}

	// Score based on how many distinct cards hit this merchant
	// 3+ cards = suspicious, 5+ = very suspicious, 10+ = almost certain ring
	if cardCount >= 10 {
		signal.RiskScore = 1.0
	} else if cardCount >= 7 {
		signal.RiskScore = 0.8
	} else if cardCount >= 5 {
		signal.RiskScore = 0.6
	} else if cardCount >= 3 {
		signal.RiskScore = 0.3
	}

	return signal, nil
}

// GetCardConnections returns merchants this card shares with other suspicious cards
func (d *Detector) GetCardConnections(ctx context.Context, cardID string) (int, error) {
	// Find all merchants this card has used recently
	cardKey := fmt.Sprintf("card:%s:merchants", cardID)
	merchants, err := d.rdb.SMembers(ctx, cardKey).Result()
	if err != nil {
		return 0, err
	}

	suspiciousConnections := 0
	for _, merchant := range merchants {
		signal, err := d.DetectRing(ctx, cardID, merchant)
		if err != nil {
			continue
		}
		if signal.RiskScore > 0.3 {
			suspiciousConnections++
		}
	}

	return suspiciousConnections, nil
}

// TrackCardMerchant records which merchants a card uses
func (d *Detector) TrackCardMerchant(ctx context.Context, cardID, merchantID string) error {
	cardKey := fmt.Sprintf("card:%s:merchants", cardID)
	pipe := d.rdb.Pipeline()
	pipe.SAdd(ctx, cardKey, merchantID)
	pipe.Expire(ctx, cardKey, 2*time.Hour)
	_, err := pipe.Exec(ctx)
	return err
}
