package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Total transactions processed
	TransactionsTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "fraud_engine_transactions_total",
		Help: "Total number of transactions processed",
	})

	// Total fraud alerts fired
	FraudAlertsTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "fraud_engine_fraud_alerts_total",
		Help: "Total number of fraud alerts published",
	})

	// Detection latency in milliseconds
	DetectionLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "fraud_engine_detection_latency_ms",
		Help:    "Detection latency in milliseconds",
		Buckets: []float64{0.1, 0.5, 1, 2, 5, 10, 20, 50},
	})

	// Current fraud score distribution
	FraudScoreHistogram = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "fraud_engine_fraud_score",
		Help:    "Distribution of fraud scores",
		Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
	})

	// Transactions processing errors
	ProcessingErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "fraud_engine_processing_errors_total",
		Help: "Total number of processing errors",
	})
)
