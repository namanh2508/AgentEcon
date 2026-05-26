// AgentEcon Chaincode - Hyperledger Fabric 2.5.x
// Prediction Market Oracle with QRE Settlement

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// Fixed-point scale for deterministic arithmetic
const (
	Scale          = 1e8 // 8 decimal places
	GridSize       = 1024
	DecayRate      = 0.95
	MinReputation  = 0.01
	MaxReputation  = 100.0
	TrimFraction   = 0.1
)

// OracleQuery represents a settlement query
type OracleQuery struct {
	QueryID      string  `json:"query_id"`
	Asset        string  `json:"asset"`
	Timestamp    int64   `json:"timestamp"`
	GroundTruth  float64 `json:"ground_truth"`
	BinID        int     `json:"bin_id"`
	RoundNumber  int     `json:"round_number"`
	Status       string  `json:"status"`
}

// ValidatorSubmission represents a validator's prediction
type ValidatorSubmission struct {
	QueryID      string  `json:"query_id"`
	ValidatorID  string  `json:"validator_id"`
	PredictedBin int     `json:"predicted_bin"`
	Timestamp    int64   `json:"timestamp"`
	Reputation   float64 `json:"reputation"`
	IsSybil      bool    `json:"is_sybil"`
}

// SettlementResult stores the result of settling a query
type SettlementResult struct {
	QueryID              string  `json:"query_id"`
	AggregatedBin        int     `json:"aggregated_bin"`
	GroundTruthBin       int     `json:"ground_truth_bin"`
	HonestReporting      bool    `json:"honest_reporting"`
	WelfareGap           float64 `json:"welfare_gap"`
	ScoringRuleReward    float64 `json:"scoring_rule_reward"`
	ValidatorsIncluded   int     `json:"validators_included"`
	SybilTrimmed         int     `json:"sybil_trimmed"`
	CommitLatencyMs      float64 `json:"commit_latency_ms"`
	SettledAt            int64   `json:"settled_at"`
}

// ReputationRecord tracks validator reputation
type ReputationRecord struct {
	ValidatorID    string    `json:"validator_id"`
	Reputation     float64   `json:"reputation"`
	AccuracyHistory []float64 `json:"accuracy_history"`
	LastUpdate     int64     `json:"last_update"`
}

// AgentEconContract implements the prediction market oracle chaincode
type AgentEconContract struct {
	contractapi.Contract
}

// Initialize sets up initial state
func (c *AgentEconContract) Initialize(ctx contractapi.TransactionContextInterface) error {
	// Initialize next round counter
	err := ctx.GetStub().PutState("next_round", []byte("1"))
	if err != nil {
		return fmt.Errorf("failed to initialize: %v", err)
	}
	return nil
}

// SubmitQuery submits a new settlement query
func (c *AgentEconContract) SubmitQuery(
	ctx contractapi.TransactionContextInterface,
	queryID string,
	asset string,
	groundTruth float64,
	groundTruthBin int,
) (*OracleQuery, error) {
	// Check if query already exists
	existing, err := ctx.GetStub().GetState(queryID)
	if err != nil {
		return nil, fmt.Errorf("failed to read state: %v", err)
	}
	if existing != nil {
		return nil, fmt.Errorf("query %s already exists", queryID)
	}

	// Get next round number
	nextRoundBytes, err := ctx.GetStub().GetState("next_round")
	if err != nil {
		return nil, fmt.Errorf("failed to get next round: %v", err)
	}
	nextRound := 1
	if nextRoundBytes != nil {
		fmt.Sscanf(string(nextRoundBytes), "%d", &nextRound)
	}

	txTime := c.txUnix(ctx)
	query := OracleQuery{
		QueryID:     queryID,
		Asset:       asset,
		Timestamp:   txTime,
		GroundTruth: groundTruth,
		BinID:       groundTruthBin,
		RoundNumber: nextRound,
		Status:      "open",
	}

	queryJSON, err := json.Marshal(query)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal query: %v", err)
	}

	err = ctx.GetStub().PutState(queryID, queryJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to put state: %v", err)
	}

	return &query, nil
}

// Submit adds a validator's prediction
func (c *AgentEconContract) Submit(
	ctx contractapi.TransactionContextInterface,
	queryID string,
	validatorID string,
	predictedBin int,
) (*ValidatorSubmission, error) {
	if predictedBin < 0 || predictedBin >= GridSize {
		return nil, fmt.Errorf("predicted bin must be in [0,%d]", GridSize-1)
	}
	queryBytes, err := ctx.GetStub().GetState(queryID)
	if err != nil {
		return nil, fmt.Errorf("failed to get query: %v", err)
	}
	if queryBytes == nil {
		return nil, fmt.Errorf("query %s not found", queryID)
	}

	// Get current reputation
	reputation, err := c.GetReputation(ctx, validatorID)
	if err != nil {
		reputation = 1.0 // Default reputation
	}

	txTime := c.txUnix(ctx)
	submission := ValidatorSubmission{
		QueryID:      queryID,
		ValidatorID:  validatorID,
		PredictedBin: predictedBin,
		Timestamp:    txTime,
		Reputation:   reputation,
		IsSybil:      false,
	}

	// Store submission with composite key
	subKey := fmt.Sprintf("%s_%s", queryID, validatorID)
	subJSON, err := json.Marshal(submission)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal submission: %v", err)
	}

	err = ctx.GetStub().PutState(subKey, subJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to put submission: %v", err)
	}

	// Add to query's submission list
	listKey := fmt.Sprintf("submissions_%s", queryID)
	listBytes, _ := ctx.GetStub().GetState(listKey)
	var subList []string
	if listBytes != nil {
		json.Unmarshal(listBytes, &subList)
	}
	found := false
	for _, existingID := range subList {
		if existingID == validatorID {
			found = true
			break
		}
	}
	if !found {
		subList = append(subList, validatorID)
	}
	listJSON, _ := json.Marshal(subList)
	ctx.GetStub().PutState(listKey, listJSON)

	return &submission, nil
}

// Settle settles a query and calculates rewards
func (c *AgentEconContract) Settle(
	ctx contractapi.TransactionContextInterface,
	queryID string,
	lambdaParam float64,
	sybilDetection bool,
) (*SettlementResult, error) {
	// Get query
	queryBytes, err := ctx.GetStub().GetState(queryID)
	if err != nil {
		return nil, fmt.Errorf("failed to get query: %v", err)
	}
	if queryBytes == nil {
		return nil, fmt.Errorf("query %s not found", queryID)
	}

	var query OracleQuery
	if err := json.Unmarshal(queryBytes, &query); err != nil {
		return nil, fmt.Errorf("failed to unmarshal query: %v", err)
	}

	// Get all submissions
	listKey := fmt.Sprintf("submissions_%s", queryID)
	listBytes, err := ctx.GetStub().GetState(listKey)
	if err != nil {
		return nil, fmt.Errorf("failed to get submission list: %v", err)
	}

	var validatorIDs []string
	if listBytes != nil {
		json.Unmarshal(listBytes, &validatorIDs)
	}

	submissions := make([]ValidatorSubmission, 0, len(validatorIDs))
	for _, vid := range validatorIDs {
		subKey := fmt.Sprintf("%s_%s", queryID, vid)
		subBytes, _ := ctx.GetStub().GetState(subKey)
		if subBytes != nil {
			var sub ValidatorSubmission
			json.Unmarshal(subBytes, &sub)
			submissions = append(submissions, sub)
		}
	}

	if len(submissions) == 0 {
		return nil, fmt.Errorf("no submissions for query %s", queryID)
	}

	// Sybil detection
	sybilTrimmed := 0
	if sybilDetection {
		sybilTrimmed = c.detectSybil(ctx, submissions)
	}

	// Filter non-Sybil submissions
	nonSybil := make([]ValidatorSubmission, 0)
	for _, s := range submissions {
		if !s.IsSybil {
			nonSybil = append(nonSybil, s)
		}
	}

	// Trim submissions
	trimmed, priceTrimmed := c.trimSubmissions(ctx, nonSybil)

	// Aggregate
	aggregatedBin := c.aggregateBins(trimmed)

	// Calculate metrics
	honestReporting := aggregatedBin == query.BinID
	welfareGap := math.Abs(float64(aggregatedBin-query.BinID)) / float64(GridSize)

	// Calculate rewards and update reputation
	totalReward := 0.0
	for i := range submissions {
		if submissions[i].IsSybil {
			continue
		}
		score := c.quadraticScoringRule(submissions[i].PredictedBin, aggregatedBin)
		newRep := c.updateReputation(ctx, submissions[i].ValidatorID, score, aggregatedBin, submissions[i].PredictedBin, lambdaParam)
		reward := score * submissions[i].Reputation
		totalReward += reward
		_ = newRep
	}

	// Create settlement result
	result := SettlementResult{
		QueryID:              queryID,
		AggregatedBin:        aggregatedBin,
		GroundTruthBin:       query.BinID,
		HonestReporting:      honestReporting,
		WelfareGap:           welfareGap,
		ScoringRuleReward:    totalReward,
		ValidatorsIncluded:   len(trimmed),
		SybilTrimmed:         sybilTrimmed + priceTrimmed,
		CommitLatencyMs:      0,
		SettledAt:            c.txUnix(ctx),
	}

	// Store settlement
	resultJSON, _ := json.Marshal(result)
	ctx.GetStub().PutState(fmt.Sprintf("settlement_%s", queryID), resultJSON)

	// Update query status
	query.Status = "settled"
	queryJSON, _ := json.Marshal(query)
	ctx.GetStub().PutState(queryID, queryJSON)

	// Increment round
	c.incrementRound(ctx)

	return &result, nil
}

// Query retrieves a query's details
func (c *AgentEconContract) Query(ctx contractapi.TransactionContextInterface, queryID string) (*OracleQuery, error) {
	queryBytes, err := ctx.GetStub().GetState(queryID)
	if err != nil {
		return nil, fmt.Errorf("failed to get query: %v", err)
	}
	if queryBytes == nil {
		return nil, fmt.Errorf("query %s not found", queryID)
	}

	var query OracleQuery
	if err := json.Unmarshal(queryBytes, &query); err != nil {
		return nil, fmt.Errorf("failed to unmarshal query: %v", err)
	}

	return &query, nil
}

// GetReputation retrieves a validator's current reputation
func (c *AgentEconContract) GetReputation(ctx contractapi.TransactionContextInterface, validatorID string) (float64, error) {
	repBytes, err := ctx.GetStub().GetState(fmt.Sprintf("rep_%s", validatorID))
	if err != nil {
		return 1.0, err
	}
	if repBytes == nil {
		return 1.0, nil
	}

	var record ReputationRecord
	if err := json.Unmarshal(repBytes, &record); err != nil {
		return 1.0, nil
	}

	return record.Reputation, nil
}

// GetSettlement retrieves settlement result for a query
func (c *AgentEconContract) GetSettlement(ctx contractapi.TransactionContextInterface, queryID string) (*SettlementResult, error) {
	resultBytes, err := ctx.GetStub().GetState(fmt.Sprintf("settlement_%s", queryID))
	if err != nil {
		return nil, fmt.Errorf("failed to get settlement: %v", err)
	}
	if resultBytes == nil {
		return nil, fmt.Errorf("settlement for %s not found", queryID)
	}

	var result SettlementResult
	if err := json.Unmarshal(resultBytes, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal settlement: %v", err)
	}

	return &result, nil
}

// GetAllSettlements returns all settlement results
func (c *AgentEconContract) GetAllSettlements(ctx contractapi.TransactionContextInterface) ([]SettlementResult, error) {
	results := make([]SettlementResult, 0)

	iterator, err := ctx.GetStub().GetStateByRange("settlement_", "settlement_~")
	if err != nil {
		return nil, err
	}
	defer iterator.Close()

	for iterator.HasNext() {
		resultBytes, _ := iterator.Next()
		var result SettlementResult
		if err := json.Unmarshal(resultBytes.GetValue(), &result); err == nil {
			results = append(results, result)
		}
	}

	return results, nil
}

// --- Helper methods ---

func (c *AgentEconContract) quadraticScoringRule(predictedBin, groundTruthBin int) float64 {
	if predictedBin == groundTruthBin {
		return 1.0
	}
	distance := math.Abs(float64(predictedBin - groundTruthBin))
	maxDistance := float64(GridSize)
	baseScore := math.Max(0.0, 1.0-math.Pow(distance/maxDistance, 2))
	return baseScore
}

func (c *AgentEconContract) updateReputation(ctx contractapi.TransactionContextInterface, validatorID string, score float64, groundTruthBin int, predictedBin int, lambdaParam float64) float64 {
	rep, _ := c.GetReputation(ctx, validatorID)

	accuracy := 0.0
	if predictedBin == groundTruthBin {
		accuracy = 1.0
	}

	reward := score * (1 + lambdaParam*accuracy)
	newRep := rep*DecayRate + reward*(1-DecayRate)
	newRep = math.Max(MinReputation, math.Min(MaxReputation, newRep))

	record := ReputationRecord{
		ValidatorID: validatorID,
		Reputation:  newRep,
		LastUpdate:  c.txUnix(ctx),
	}
	repJSON, _ := json.Marshal(record)
	ctx.GetStub().PutState(fmt.Sprintf("rep_%s", validatorID), repJSON)

	return newRep
}

func (c *AgentEconContract) detectSybil(ctx contractapi.TransactionContextInterface, submissions []ValidatorSubmission) int {
	trimmedCount := 0

	// Simple heuristic: validators with reputation below threshold
	for i := range submissions {
		rep, _ := c.GetReputation(ctx, submissions[i].ValidatorID)
		if rep < 0.1 {
			submissions[i].IsSybil = true
			trimmedCount++
		}
	}

	// Detect coordinated behavior (same prediction from many low-rep validators)
	binCount := make(map[int]int)
	for _, s := range submissions {
		if !s.IsSybil {
			binCount[s.PredictedBin]++
		}
	}

	for bin, count := range binCount {
		if count >= 5 {
			for i := range submissions {
				if submissions[i].PredictedBin == bin && !submissions[i].IsSybil {
					rep, _ := c.GetReputation(ctx, submissions[i].ValidatorID)
					if rep < 0.2 {
						submissions[i].IsSybil = true
						trimmedCount++
					}
				}
			}
		}
	}

	return trimmedCount
}

func (c *AgentEconContract) trimSubmissions(ctx contractapi.TransactionContextInterface, submissions []ValidatorSubmission) ([]ValidatorSubmission, int) {
	if len(submissions) == 0 {
		return submissions, 0
	}

	nTrim := int(float64(len(submissions)) * TrimFraction)
	if nTrim == 0 || 2*nTrim >= len(submissions) {
		return submissions, 0
	}

	sort.Slice(submissions, func(i, j int) bool {
		if submissions[i].PredictedBin == submissions[j].PredictedBin {
			return submissions[i].ValidatorID < submissions[j].ValidatorID
		}
		return submissions[i].PredictedBin < submissions[j].PredictedBin
	})

	return submissions[nTrim : len(submissions)-nTrim], 2 * nTrim
}

func (c *AgentEconContract) aggregateBins(submissions []ValidatorSubmission) int {
	if len(submissions) == 0 {
		return GridSize / 2
	}

	totalWeight := 0.0
	weightedSum := 0.0

	for _, s := range submissions {
		weightedSum += float64(s.PredictedBin) * s.Reputation
		totalWeight += s.Reputation
	}

	if totalWeight == 0 {
		return GridSize / 2
	}

	meanBin := weightedSum / totalWeight
	result := int(math.Round(meanBin))

	if result < 0 {
		return 0
	}
	if result >= GridSize {
		return GridSize - 1
	}
	return result
}

func (c *AgentEconContract) txUnix(ctx contractapi.TransactionContextInterface) int64 {
	ts, err := ctx.GetStub().GetTxTimestamp()
	if err != nil || ts == nil {
		return 0
	}
	return ts.Seconds
}

func (c *AgentEconContract) incrementRound(ctx contractapi.TransactionContextInterface) {
	nextRoundBytes, _ := ctx.GetStub().GetState("next_round")
	nextRound := 1
	if nextRoundBytes != nil {
		fmt.Sscanf(string(nextRoundBytes), "%d", &nextRound)
	}
	nextRound++
	ctx.GetStub().PutState("next_round", []byte(fmt.Sprintf("%d", nextRound)))
}

func main() {
	chaincode, err := contractapi.NewChaincode(&AgentEconContract{})
	if err != nil {
		fmt.Printf("Error creating chaincode: %v", err)
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting chaincode: %v", err)
	}
}
