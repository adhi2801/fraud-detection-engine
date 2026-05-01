package publisher

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"fraud-engine/shared"

	"github.com/twmb/franz-go/pkg/kgo"
)

const alertTopic = "fraud-alerts"

type Publisher struct {
	client *kgo.Client
}

func New(brokerAddr string) (*Publisher, error) {
	client, err := kgo.NewClient(
		kgo.SeedBrokers(brokerAddr),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create publisher: %w", err)
	}
	return &Publisher{client: client}, nil
}

func (p *Publisher) Publish(ctx context.Context, alert shared.FraudAlert) error {
	payload, err := json.Marshal(alert)
	if err != nil {
		return fmt.Errorf("marshal alert: %w", err)
	}

	record := &kgo.Record{
		Topic: alertTopic,
		Key:   []byte(alert.CardID),
		Value: payload,
	}

	if err := p.client.ProduceSync(ctx, record).FirstErr(); err != nil {
		return fmt.Errorf("publish alert: %w", err)
	}

	log.Printf("🚨 FRAUD ALERT | card: %s | score: %.2f | tx: %s",
		alert.CardID, alert.FraudScore, alert.TransactionID)

	return nil
}

func (p *Publisher) Close() {
	p.client.Close()
}
