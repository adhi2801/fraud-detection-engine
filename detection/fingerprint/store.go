package fingerprint

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"fraud-engine/shared"

	"github.com/redis/go-redis/v9"
)

// CardProfile stores permanent behavioral history for a card
type CardProfile struct {
	CardID      string    `json:"card_id"`
	TxnCount    int64     `json:"txn_count"`
	TotalSpend  float64   `json:"total_spend"`
	AvgSpend    float64   `json:"avg_spend"`
	MaxSpend    float64   `json:"max_spend"`
	Countries   []string  `json:"countries"`
	ActiveHours []int     `json:"active_hours"`
	LastUpdated time.Time `json:"last_updated"`
}

// AnomalyScore represents how different a transaction is from card's profile
type AnomalyScore struct {
	AmountAnomaly  float32 `json:"amount_anomaly"`
	CountryAnomaly float32 `json:"country_anomaly"`
	HourAnomaly    float32 `json:"hour_anomaly"`
	OverallAnomaly float32 `json:"overall_anomaly"`
}

type Store struct {
	rdb *redis.Client
}

func New(rdb *redis.Client) *Store {
	return &Store{rdb: rdb}
}

func profileKey(cardID string) string {
	return fmt.Sprintf("profile:%s", cardID)
}

// GetProfile retrieves card profile from Redis
func (s *Store) GetProfile(ctx context.Context, cardID string) (*CardProfile, error) {
	data, err := s.rdb.Get(ctx, profileKey(cardID)).Bytes()
	if err == redis.Nil {
		return nil, nil // no profile yet
	}
	if err != nil {
		return nil, err
	}

	var profile CardProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return nil, err
	}
	return &profile, nil
}

// UpdateProfile updates card profile with new transaction
func (s *Store) UpdateProfile(ctx context.Context, tx shared.Transaction) error {
	profile, err := s.GetProfile(ctx, tx.CardID)
	if err != nil {
		return err
	}

	if profile == nil {
		profile = &CardProfile{
			CardID:   tx.CardID,
			MaxSpend: tx.AmountUSD,
		}
	}

	// Update running statistics
	profile.TxnCount++
	profile.TotalSpend += tx.AmountUSD
	profile.AvgSpend = profile.TotalSpend / float64(profile.TxnCount)

	if tx.AmountUSD > profile.MaxSpend {
		profile.MaxSpend = tx.AmountUSD
	}

	// Track countries
	if !contains(profile.Countries, tx.CountryCode) {
		profile.Countries = append(profile.Countries, tx.CountryCode)
	}

	// Track active hours
	hour := tx.Timestamp.Hour()
	if !containsInt(profile.ActiveHours, hour) {
		profile.ActiveHours = append(profile.ActiveHours, hour)
	}

	profile.LastUpdated = time.Now()

	data, err := json.Marshal(profile)
	if err != nil {
		return err
	}

	// Store permanently — no expiry for behavioral profiles
	return s.rdb.Set(ctx, profileKey(tx.CardID), data, 0).Err()
}

// ScoreAnomaly compares transaction against card's behavioral profile
func (s *Store) ScoreAnomaly(ctx context.Context, tx shared.Transaction) (AnomalyScore, error) {
	profile, err := s.GetProfile(ctx, tx.CardID)
	if err != nil || profile == nil || profile.TxnCount < 5 {
		// Not enough history yet — return neutral score
		return AnomalyScore{}, nil
	}

	var score AnomalyScore

	// Amount anomaly — how many standard deviations above average?
	if profile.AvgSpend > 0 {
		ratio := tx.AmountUSD / profile.AvgSpend
		if ratio > 10 {
			score.AmountAnomaly = 1.0
		} else if ratio > 5 {
			score.AmountAnomaly = 0.8
		} else if ratio > 3 {
			score.AmountAnomaly = 0.5
		} else if ratio > 2 {
			score.AmountAnomaly = 0.3
		}
	}

	// Country anomaly — is this a new country for this card?
	if !contains(profile.Countries, tx.CountryCode) {
		score.CountryAnomaly = 1.0
	}

	// Hour anomaly — is this an unusual hour for this card?
	hour := tx.Timestamp.Hour()
	if !containsInt(profile.ActiveHours, hour) {
		score.HourAnomaly = 0.5
	}

	// Overall anomaly — weighted combination
	score.OverallAnomaly = float32(math.Min(1.0,
		float64(score.AmountAnomaly*0.5+
			score.CountryAnomaly*0.3+
			score.HourAnomaly*0.2)))

	return score, nil
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func containsInt(slice []int, item int) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
