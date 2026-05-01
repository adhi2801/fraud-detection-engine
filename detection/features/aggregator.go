package features

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"fraud-engine/shared"

	"github.com/redis/go-redis/v9"
)

const windowDuration = time.Hour

type Aggregator struct {
	rdb *redis.Client
}

func New(rdb *redis.Client) *Aggregator {
	return &Aggregator{rdb: rdb}
}

func (a *Aggregator) Aggregate(ctx context.Context, tx shared.Transaction) (shared.Features, error) {
	now := time.Now().UnixMilli()
	windowStart := float64(now - windowDuration.Milliseconds())
	score := float64(now)
	key := fmt.Sprintf("card:%s:txns", tx.CardID)
	member := fmt.Sprintf("%s:%.2f:%s", tx.ID, tx.AmountUSD, tx.MerchantID)

	pipe := a.rdb.Pipeline()
	pipe.ZAdd(ctx, key, redis.Z{Score: score, Member: member})
	pipe.Expire(ctx, key, windowDuration+time.Minute)
	zCardCmd := pipe.ZCard(ctx, key)
	rangeCmd := pipe.ZRangeByScore(ctx, key, &redis.ZRangeBy{
		Min: strconv.FormatFloat(windowStart, 'f', -1, 64),
		Max: "+inf",
	})

	if _, err := pipe.Exec(ctx); err != nil {
		return shared.Features{}, fmt.Errorf("redis pipeline: %w", err)
	}

	var totalSpend float64
	distinctMerch := map[string]struct{}{}

	for _, m := range rangeCmd.Val() {
		parts := strings.SplitN(m, ":", 3)
		if len(parts) == 3 {
			if amt, err := strconv.ParseFloat(parts[1], 64); err == nil {
				totalSpend += amt
			}
			distinctMerch[parts[2]] = struct{}{}
		}
	}

	isIntl := float32(0)
	if tx.CountryCode != "US" {
		isIntl = 1
	}

	return shared.Features{
		AmountUSD:       float32(tx.AmountUSD),
		TxnCount1h:      float32(zCardCmd.Val()),
		TotalSpend1h:    float32(totalSpend),
		DistinctMerch1h: float32(len(distinctMerch)),
		IsInternational: isIntl,
	}, nil
}
