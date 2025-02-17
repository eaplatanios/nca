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

import Foundation
import TensorFlow

/// Tokenized text passage.
public struct TextBatch: KeyPathIterable {
  /// IDs that correspond to the vocabulary used while tokenizing.
  /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
  public var tokenIds: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.

  /// IDs of the token types (e.g., sentence A and sentence B in BERT).
  /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
  public var tokenTypeIds: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.

  /// Mask over the sequence of tokens specifying which ones are "real" as opposed to "padding".
  /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
  public var mask: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.
}

/// Tokenized text passage.
public struct TokenizedText {
  /// IDs that correspond to the vocabulary used while tokenizing.
  public let tokenIds: [Int]

  /// IDs of the token types (e.g., sentence A and sentence B in BERT).
  public let tokenTypeIds: [Int]

  /// Mask over the sequence of tokens specifying which ones are "real" as opposed to "padding".
  public let mask: [Bool]
}

/// Returns a 3-D attention mask that correspond to the 2-D mask of the provided text batch.
///
/// - Parameters:
///   - text: Text batch for which to create an attention mask. `input.mask` has shape
///     `[batchSize, sequenceLength]`.
///
/// - Returns: Attention mask with shape `[batchSize, sequenceLength, sequenceLength]`.
public func createAttentionMask(forTextBatch text: TextBatch) -> Tensor<Float> {
  let batchSize = text.tokenIds.shape[0]
  let fromSequenceLength = text.tokenIds.shape[1]
  let toSequenceLength = text.mask.shape[1]
  let reshapedMask = Tensor<Float>(text.mask.reshaped(to: [batchSize, 1, toSequenceLength]))

  // We do not assume that `input.tokenIds` is a mask. We do not actually care if we attend
  // *from* padding tokens (only *to* padding tokens) so we create a tensor of all ones.
  let broadcastOnes = Tensor<Float>(ones: [batchSize, fromSequenceLength, 1])

  // We broadcast along two dimensions to create the mask.
  return broadcastOnes * reshapedMask
}

/// Preprocesses an array of text sequences and prepares them for use by a text perception
/// module. Preprocessing mainly consists of tokenization.
///
/// - Parameters:
///   - sequences: Text sequences (not tokenized).
///   - maxSequenceLength: Maximum sequence length supported by the text perception module. This
///     is mainly used for padding the preprocessed sequences.
///   - tokenizer: Tokenizer to use while preprocessing.
///
/// - Returns: Tokenized text.
public func preprocessText(
  sequences: [String],
  maxSequenceLength: Int,
  usingTokenizer tokenizer: FullTextTokenizer
) -> TokenizedText {
  var sequences = sequences.map(tokenizer.tokenize)

  // Truncate the sequences based on the maximum allowed sequence length, while accounting for
  // the '[CLS]' token and for `sequences.count` '[SEP]' tokens. The following is a simple
  // heuristic which will truncate the longer sequence one token at a time. This makes more sense
  // than truncating an equal percent of tokens from each sequence, since if one sequence is very
  // short then each token that is truncated likely contains more information than respective
  // tokens in longer sequences.
  var totalLength = sequences.map { $0.count }.reduce(0, +)
  while totalLength >= maxSequenceLength - 1 - sequences.count {
    let maxIndex = sequences.enumerated().max(by: { $0.1.count < $1.1.count })!.0
    sequences[maxIndex] = [String](sequences[maxIndex].dropLast())
    totalLength = sequences.map { $0.count }.reduce(0, +)
  }

  // The convention in BERT is:
  //   (a) For sequence pairs:
  //       tokens:       [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  //       tokenTypeIds: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  //   (b) For single sequences:
  //       tokens:       [CLS] the dog is hairy . [SEP]
  //       tokenTypeIds: 0     0   0   0  0     0 0
  // where "tokenTypeIds" are used to indicate whether this is the first sequence or the second
  // sequence. The embedding vectors for `tokenTypeId = 0` and `tokenTypeId = 1` were learned
  // during pre-training and are added to the WordPiece embedding vector (and position vector).
  // This is not *strictly* necessary since the [SEP] token unambiguously separates the
  // sequences. However, it makes it easier for the model to learn the concept of sequences.
  //
  // For classification tasks, the first vector (corresponding to `[CLS]`) is used as the
  // "sentence embedding". Note that this only makes sense because the entire model is fine-tuned
  // under this assumption.
  var tokens = ["[CLS]"]
  var tokenTypeIds = [0]
  for (sequenceId, sequence) in sequences.enumerated() {
    for token in sequence {
      tokens.append(token)
      tokenTypeIds.append(sequenceId)
    }
    tokens.append("[SEP]")
    tokenTypeIds.append(sequenceId)
  }
  let tokenIds = tokens.map { tokenizer.vocabulary.tokensToIds[$0]! }

  // The mask is set to `true` for real tokens and `false` for padding tokens. This is so that
  // only real tokens are attended to.
  let mask = [Bool](repeating: true, count: tokenIds.count)

  return TokenizedText(tokenIds: tokenIds, tokenTypeIds: tokenTypeIds, mask: mask)
}

// TODO: !!! Add documentation.
public func padAndBatch(textBatches: [TextBatch]) -> TextBatch {
  let maxLength = textBatches.map { $0.tokenIds.shape[0] }.max()!
  let paddedBatches = textBatches.map { batch -> TextBatch in
    let paddingSize = maxLength - batch.tokenIds.shape[0]
    return TextBatch(
      tokenIds: batch.tokenIds.padded(forSizes: [(before: 0, after: paddingSize)]),
      tokenTypeIds: batch.tokenTypeIds.padded(forSizes: [(before: 0, after: paddingSize)]),
      mask: batch.mask.padded(forSizes: [(before: 0, after: paddingSize)]))
  }
  return TextBatch(
    tokenIds: Tensor<Int32>(stacking: paddedBatches.map { $0.tokenIds }, alongAxis: 0),
    tokenTypeIds: Tensor<Int32>(stacking: paddedBatches.map { $0.tokenTypeIds }, alongAxis: 0),
    mask: Tensor<Int32>(stacking: paddedBatches.map { $0.mask }, alongAxis: 0))
}

/// Vocabulary that can be used for tokenizing strings.
public struct Vocabulary {
  internal let tokensToIds: [String: Int]
  internal let idsToTokens: [Int: String]

  public var count: Int { tokensToIds.count }

  public init(tokensToIds: [String: Int]) {
    self.tokensToIds = tokensToIds
    self.idsToTokens = [Int: String](uniqueKeysWithValues: tokensToIds.map { ($1, $0) })
  }

  public init(idsToTokens: [Int: String]) {
    self.tokensToIds = [String: Int](uniqueKeysWithValues: idsToTokens.map { ($1, $0) })
    self.idsToTokens = idsToTokens
  }

  public func contains(token: String) -> Bool {
    tokensToIds.keys.contains(token)
  }

  public func id(forToken token: String) -> Int? {
    tokensToIds[token]
  }

  public func token(forId id: Int) -> String? {
    idsToTokens[id]
  }
}

extension Vocabulary: Serializable {
  public init(fromFile fileURL: URL) throws {
    self.init(
      tokensToIds: [String: Int](
        (try String(contentsOfFile: fileURL.path, encoding: .utf8))
          .components(separatedBy: .newlines)
          .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
          .filter { $0.count > 0 }
          .enumerated().map { ($0.element, $0.offset) },
        uniquingKeysWith: { (v1, v2) in max(v1, v2) }))
  }

  public func save(toFile fileURL: URL) throws {
    try idsToTokens
      .sorted { $0.key < $1.key }
      .map { $0.1 }
      .joined(separator: "\n")
      .write(to: fileURL, atomically: true, encoding: .utf8)
  }
}

extension Vocabulary {
  public init(fromSentencePieceModel fileURL: URL) throws {
    self.init(
      tokensToIds: [String: Int](
        (try Sentencepiece_ModelProto(serializedData: Data(contentsOf: fileURL)))
          .pieces
          .map { $0.piece.replacingOccurrences(of: "▁", with: "##") }
          .map { $0 == "<unk>" ? "[UNK]" : $0 }
          .enumerated().map { ($0.element, $0.offset) },
        uniquingKeysWith: { (v1, v2) in max(v1, v2) }))
  }
}

/// Text tokenizer which is used to split strings into arrays of tokens.
public protocol TextTokenizer {
  func tokenize(_ text: String) -> [String]
}

/// Basic text tokenizer that performs some simple preprocessing to clean the provided text and
/// then performs tokenization based on whitespaces.
public struct BasicTextTokenizer: TextTokenizer {
  public let caseSensitive: Bool

  /// Creates a basic text tokenizer.
  ///
  /// Arguments:
  ///   - caseSensitive: Specifies whether or not to ignore case.
  public init(caseSensitive: Bool = false) {
    self.caseSensitive = caseSensitive
  }

  public func tokenize(_ text: String) -> [String] {
    clean(text).split(separator: " ").flatMap { token -> [String] in
      var processed = String(token)
      if !caseSensitive {
        processed = processed.lowercased()

        // Normalize unicode characters.
        processed = processed.decomposedStringWithCanonicalMapping

        // Strip accents.
        processed = processed.replacingOccurrences(
          of: #"\p{Mn}"#,
          with: "",
          options: .regularExpression)
      }

      // Split punctuation. We treat all non-letter/number ASCII as punctuation. Characters such as
      // "$" are not in the Unicode Punctuation class but we treat them as punctuation anyways
      // for consistency.
      processed = processed.replacingOccurrences(
        of: #"([\p{P}!-/:-@\[-`{-~])"#,
        with: " $1 ",
        options: .regularExpression)

      return processed.split(separator: " ").map(String.init)
    }
  }
}

/// Subword tokenizer.
///
/// This tokenizer uses a greedy longest-match-first algorithm to perform tokenization using the
/// provided vocabulary. For example, `"unaffable"` could be tokenized as
/// `["un", "##aff", "##able"]`.
public struct SubwordTokenizer: TextTokenizer {
  public let vocabulary: Vocabulary
  public let unknownToken: String
  public let maxTokenLength: Int

  /// Creates a subword tokenizer.
  ///
  /// - Parameters:
  ///   - vocabulary: Vocabulary containing all supported tokens.
  ///   - unknownToken: Token used to represent unknown tokens (i.e., tokens that are not in the
  ///     provided vocabulary or whose length is longer than `maxTokenLength`).
  ///   - maxTokenLength: Maximum allowed token length.
  public init(vocabulary: Vocabulary, unknownToken: String = "[UNK]", maxTokenLength: Int) {
    self.vocabulary = vocabulary
    self.unknownToken = unknownToken
    self.maxTokenLength = maxTokenLength
  }

  public func tokenize(_ text: String) -> [String] {
    clean(text).split(separator: " ").flatMap { token -> [String] in
      if token.count > maxTokenLength { return [unknownToken] }
      var isBad = false
      var start = token.startIndex
      var subTokens = [String]()
      while start < token.endIndex {
        // Find the longest matching substring.
        var end = token.endIndex
        var currentSubstring = ""
        while start < end {
          var substring = String(token[start..<end])
          if start > token.startIndex {
            substring = "##" + substring
          }
          if vocabulary.contains(token: substring) {
            currentSubstring = substring
            start = end
          } else {
            end = token.index(end, offsetBy: -1)
          }
        }

        // Check if the substring is good.
        if currentSubstring.isEmpty {
          isBad = true
          start = token.endIndex
        } else {
          subTokens.append(currentSubstring)
          start = end
        }
      }
      return isBad ? [unknownToken] : subTokens
    }
  }
}

/// Full text tokenizer that is simply defined as the composition of the basic text tokenizer and
/// the subword tokenizer.
public struct FullTextTokenizer: TextTokenizer {
  public let caseSensitive: Bool
  public let vocabulary: Vocabulary
  public let unknownToken: String
  public let maxTokenLength: Int

  private let basicTextTokenizer: BasicTextTokenizer
  private let subwordTokenizer: SubwordTokenizer

  /// Creates a full text tokenizer.
  ///
  /// - Parameters:
  ///   - caseSensitive: Specifies whether or not to ignore case.
  ///   - vocabulary: Vocabulary containing all supported tokens.
  ///   - unknownToken: Token used to represent unknown tokens (i.e., tokens that are not in the
  ///     provided vocabulary or whose length is longer than `maxTokenLength`).
  ///   - maxTokenLength: Maximum allowed token length.
  public init(
    caseSensitive: Bool = false,
    vocabulary: Vocabulary,
    unknownToken: String = "[UNK]",
    maxTokenLength: Int = 200
  ) {
    self.caseSensitive = caseSensitive
    self.vocabulary = vocabulary
    self.unknownToken = unknownToken
    self.maxTokenLength = maxTokenLength
    self.basicTextTokenizer = BasicTextTokenizer(caseSensitive: caseSensitive)
    self.subwordTokenizer = SubwordTokenizer(
      vocabulary: vocabulary,
      unknownToken: unknownToken,
      maxTokenLength: maxTokenLength)
  }

  public func tokenize(_ text: String) -> [String] {
    basicTextTokenizer.tokenize(text).flatMap(subwordTokenizer.tokenize)
  }
}

/// Returns a cleaned version of the provided string. Cleaning in this case consists of normalizing
/// whitespaces, removing control characters and adding whitespaces around CJK characters.
///
/// - Parameters:
///   - text: String to clean.
///
/// - Returns: Cleaned version of `text`.
internal func clean(_ text: String) -> String {
  // Normalize whitespaces.
  let afterWhitespace = text.replacingOccurrences(
    of: #"\s+"#,
    with: " ",
    options: .regularExpression)

  // Remove control characters.
  let afterControl = afterWhitespace.replacingOccurrences(
    of: #"[\x{0000}\x{fffd}\p{C}]"#,
    with: "",
    options: .regularExpression)

  // Add whitespace around CJK characters.
  //
  // The regular expression that we use defines a "chinese character" as anything in the
  // [CJK Unicode block](https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)).
  //
  // Note that the CJK Unicode block is not all Japanese and Korean characters, despite its name.
  // The modern Korean Hangul alphabet is a different block, as is Japanese Hiragana and
  // Katakana. Those alphabets are used to write space-separated words, and so they are not
  // treated specially and are instead handled like all of the other languages.
  let afterCJK = afterControl.replacingOccurrences(
    of: #"([\p{InCJK_Unified_Ideographs}"# +
      #"\p{InCJK_Unified_Ideographs_Extension_A}"# +
      #"\p{InCJK_Compatibility_Ideographs}"# +
      #"\x{20000}-\x{2a6df}"# +
      #"\x{2a700}-\x{2b73f}"# +
      #"\x{2b740}-\x{2b81f}"# +
      #"\x{2b820}-\x{2ceaf}"# +
      #"\x{2f800}-\x{2fa1f}])"#,
    with: " $1 ",
    options: .regularExpression)

  return afterCJK
}
