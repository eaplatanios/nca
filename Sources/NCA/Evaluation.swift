// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

public func accuracy<T: Equatable>(predictions: [T], groundTruth: [T]) -> Float {
  let nominator = Float(zip(predictions, groundTruth).map { $0 == $1 ? 1 : 0 }.reduce(0, +))
  let denominator = Float(predictions.count)
  return nominator / denominator
}

/// Computes the F1 score.
public func f1Score(predictions: [Bool], groundTruth: [Bool]) -> Float {
  var tp = 0 // True positives.
  var tn = 0 // True negatives.
  var fp = 0 // False positives.
  var fn = 0 // False negatives.
  for (prediction, truth) in zip(predictions, groundTruth) {
    switch (prediction, truth) {
    case (false, false): tn += 1
    case (false, true): fn += 1
    case (true, false): fp += 1
    case (true, true): tp += 1
    }
  }
  let precision = tp + fp > 0 ? Float(tp) / Float(tp + fp) : 1
  let recall = tp + fn > 0 ? Float(tp) / Float(tp + fn) : 1
  let nominator = precision * recall
  let denominator = precision + recall
  return denominator == 0.0 ? 0.0 : 2 * nominator / denominator
}

/// Computes the Matthews correlation coefficient.
///
/// The Matthews correlation coefficient is more informative than other confusion matrix measures
/// (such as F1 score and accuracy) in evaluating binary classification problems, because it takes
/// into account the balance ratios of the four confusion matrix categories (true positives, true
/// negatives, false positives, false negatives).
///
/// - Source: [https://en.wikipedia.org/wiki/Matthews_correlation_coefficient](
///             https://en.wikipedia.org/wiki/Matthews_correlation_coefficient).
public func matthewsCorrelationCoefficient(predictions: [Bool], groundTruth: [Bool]) -> Float {
  var tp = 0 // True positives.
  var tn = 0 // True negatives.
  var fp = 0 // False positives.
  var fn = 0 // False negatives.
  for (prediction, truth) in zip(predictions, groundTruth) {
    switch (prediction, truth) {
    case (false, false): tn += 1
    case (false, true): fn += 1
    case (true, false): fp += 1
    case (true, true): tp += 1
    }
  }
  let nominator = Float(tp * tn - fp * fn)
  let denominator = Float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).squareRoot()
  return denominator != 0 ? nominator / denominator : 0
}

/// Computes the Pearson correlation coefficient.
///
/// - Source: [https://en.wikipedia.org/wiki/Pearson_correlation_coefficient](
///             https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
public func pearsonCorrelationCoefficient(predictions: [Float], groundTruth: [Float]) -> Float {
  let pMean = predictions.reduce(0, +) / Float(predictions.count)
  let tMean = groundTruth.reduce(0, +) / Float(groundTruth.count)
  let nominator = zip(predictions, groundTruth).map { ($0 - pMean) * ($1 - tMean) }.reduce(0, +)
  let pDenominator = (predictions.map { ($0 - pMean) * ($0 - pMean) }).reduce(0, +).squareRoot()
  let tDenominator = (groundTruth.map { ($0 - tMean) * ($0 - tMean) }).reduce(0, +).squareRoot()
  return nominator / (pDenominator * tDenominator)
}

/// Computes the Spearman correlation coefficient.
///
/// - Source: [https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient](
///             https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient).
public func spearmanCorrelationCoefficient(predictions: [Float], groundTruth: [Float]) -> Float {
  let pRanked = predictions.ranked()
  let tRanked = groundTruth.ranked()
  let differences = zip(pRanked, tRanked).map { ($0 - $1) * ($0 - $1) }.reduce(0, +)
  let n = Float(predictions.count)
  let denominator = n * (n * n - 1)
  return 1 - 6 * differences / denominator
}

extension Array where Element == Float {
  /// Returns the rank of each element in this array.
  ///
  /// - Note: Ties are broken by averaging the ranks of the corresponding elements.
  internal func ranked() -> [Float] {
    let sorted = self.enumerated().sorted(by: { $0.1 < $1.1 })
    var ranks = Array((0..<count).map(Float.init))
    var rank = 1
    var n = 1
    var i = 0
    while i < count {
      var j = i

      // Get the number of elements with equal rank.
      while j < count - 1 && sorted[j].1 == sorted[j + 1].1 { j += 1 }
      n = j - i + 1

      // Compute and assign the rank.
      for j in 0..<n { ranks[sorted[i + j].0] = Float(rank) + Float(n - 1) * 0.5 }

      rank += n
      i += n
    }
    return ranks
  }
}
