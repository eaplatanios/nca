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

/// BERT layer for encoding text.
///
/// - Sources:
///   - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](
///       https://arxiv.org/pdf/1810.04805.pdf).
///   - [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations](
///       https://arxiv.org/pdf/1909.11942.pdf).
public struct BERT: TextPerceptionModule { // <Scalar: TensorFlowFloatingPoint & Codable>: Module {
  // TODO: !!! Convert to a generic constraint once TF-427 is resolved.
  public typealias Scalar = Float

  @noDerivative public let variant: Variant
  @noDerivative public let vocabulary: Vocabulary
  @noDerivative public let caseSensitive: Bool
  @noDerivative public let tokenizer: TextTokenizer
  @noDerivative public let hiddenSize: Int
  @noDerivative public let hiddenLayerCount: Int
  @noDerivative public let attentionHeadCount: Int
  @noDerivative public let intermediateSize: Int
  @noDerivative public let intermediateActivation: Activation<Scalar>
  @noDerivative public let hiddenDropoutProbability: Scalar
  @noDerivative public let attentionDropoutProbability: Scalar
  @noDerivative public let maxSequenceLength: Int
  @noDerivative public let typeVocabularySize: Int
  @noDerivative public let initializerStandardDeviation: Scalar

  public var tokenEmbedding: Embedding<Scalar>
  public var tokenTypeEmbedding: Embedding<Scalar>
  public var positionEmbedding: Embedding<Scalar>
  public var embeddingLayerNormalization: LayerNormalization<Scalar>
  @noDerivative public var embeddingDropout: Dropout<Scalar>
  public var embeddingProjection: [Affine<Scalar>] // TODO: [AD] Change to optional once supported.
  public var encoderLayers: [TransformerEncoderLayer]

  public var regularizationValue: TangentVector {
    TangentVector(
      tokenEmbedding: tokenEmbedding.regularizationValue,
      tokenTypeEmbedding: tokenTypeEmbedding.regularizationValue,
      positionEmbedding: positionEmbedding.regularizationValue,
      embeddingLayerNormalization: embeddingLayerNormalization.regularizationValue,
      embeddingProjection: [Affine<Scalar>].TangentVector(
        embeddingProjection.map { $0.regularizationValue }),
      encoderLayers: [TransformerEncoderLayer].TangentVector(
        encoderLayers.map { $0.regularizationValue }))
  }

  /// TODO: [DOC] Add a documentation string and fix the parameter descriptions.
  ///
  /// - Parameters:
  ///   - hiddenSize: Size of the encoder and the pooling layers.
  ///   - hiddenLayerCount: Number of hidden layers in the encoder.
  ///   - attentionHeadCount: Number of attention heads for each encoder attention layer.
  ///   - intermediateSize: Size of the encoder "intermediate" (i.e., feed-forward) layer.
  ///   - intermediateActivation: Activation function used in the encoder and the pooling layers.
  ///   - hiddenDropoutProbability: Dropout probability for all fully connected layers in the
  ///     embeddings, the encoder, and the pooling layers.
  ///   - attentionDropoutProbability: Dropout probability for the attention scores.
  ///   - maxSequenceLength: Maximum sequence length that this model might ever be used with.
  ///     Typically, this is set to something large, just in case (e.g., 512, 1024, or 2048).
  ///   - typeVocabularySize: Vocabulary size for the token type IDs passed into the BERT model.
  ///   - initializerStandardDeviation: Standard deviation of the truncated Normal initializer
  ///     used for initializing all weight matrices.
  public init(
    variant: Variant,
    vocabulary: Vocabulary,
    caseSensitive: Bool,
    hiddenSize: Int = 768,
    hiddenLayerCount: Int = 12,
    attentionHeadCount: Int = 12,
    intermediateSize: Int = 3072,
    intermediateActivation: @escaping Activation<Scalar> = gelu,
    hiddenDropoutProbability: Scalar = 0.1,
    attentionDropoutProbability: Scalar = 0.1,
    maxSequenceLength: Int = 512,
    typeVocabularySize: Int = 2,
    initializerStandardDeviation: Scalar = 0.02,
    useOneHotEmbeddings: Bool = false
  ) {
    self.variant = variant
    self.vocabulary = vocabulary
    self.caseSensitive = caseSensitive
    self.hiddenSize = hiddenSize
    self.hiddenLayerCount = hiddenLayerCount
    self.attentionHeadCount = attentionHeadCount
    self.intermediateSize = intermediateSize
    self.intermediateActivation = intermediateActivation
    self.hiddenDropoutProbability = hiddenDropoutProbability
    self.attentionDropoutProbability = attentionDropoutProbability
    self.maxSequenceLength = maxSequenceLength
    self.typeVocabularySize = typeVocabularySize
    self.initializerStandardDeviation = initializerStandardDeviation
    self.tokenizer = FullTextTokenizer(
      caseSensitive: caseSensitive,
      vocabulary: vocabulary,
      unknownToken: "[UNK]",
      maxTokenLength: nil)

    if case let .albert(_, hiddenGroupCount) = variant {
      precondition(
        hiddenGroupCount <= hiddenLayerCount,
        "The number of hidden groups must not be greater than the number of hidden layers.")
    }

    let embeddingSize: Int = {
      switch variant {
      case .originalBert: return hiddenSize
      case let .albert(embeddingSize, _): return embeddingSize
      }
    }()

    self.tokenEmbedding = Embedding<Scalar>(
      vocabularySize: vocabulary.count,
      embeddingSize: embeddingSize,
      embeddingsInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor<Scalar>(initializerStandardDeviation)),
      useOneHotEmbeddings: useOneHotEmbeddings)

    // The token type vocabulary will always be small and so we use the one-hot approach here as
    // it is always faster for small vocabularies.
    self.tokenTypeEmbedding = Embedding<Scalar>(
      vocabularySize: typeVocabularySize,
      embeddingSize: embeddingSize,
      embeddingsInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor<Scalar>(initializerStandardDeviation)),
      useOneHotEmbeddings: true)

    // Since the position embeddings table is a learned variable, we create it using a (long)
    // sequence length, `maxSequenceLength`. The actual sequence length might be shorter than this,
    // for faster training of tasks that do not have long sequences. So, `positionEmbedding`
    // effectively contains an embedding table for positions
    // [0, 1, 2, ..., maxPositionEmbeddings - 1], and the current sequence may have positions
    // [0, 1, 2, ..., sequenceLength - 1], so we can just perform a slice.
    self.positionEmbedding = Embedding(
      vocabularySize: maxSequenceLength,
      embeddingSize: embeddingSize,
      embeddingsInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(initializerStandardDeviation)),
      useOneHotEmbeddings: false)

    self.embeddingLayerNormalization = LayerNormalization<Scalar>(
      featureCount: hiddenSize,
      axis: -1)
    self.embeddingDropout = Dropout(probability: hiddenDropoutProbability)

    // Add an embedding projection layer if using the ALBERT variant.
    self.embeddingProjection = {
      switch variant {
      case .originalBert: return []
      case let .albert(embeddingSize, _):
        // TODO: [AD] Change to optional once supported.
        return [Affine<Scalar>(
          inputSize: embeddingSize,
          outputSize: hiddenSize,
          weightInitializer: truncatedNormalInitializer(
            standardDeviation: Tensor(initializerStandardDeviation)))]
      }
    }()

    switch variant {
    case .originalBert:
      self.encoderLayers = (0..<hiddenLayerCount).map { _ in
        TransformerEncoderLayer(
          hiddenSize: hiddenSize,
          attentionHeadCount: attentionHeadCount,
          attentionQueryActivation: { $0 },
          attentionKeyActivation: { $0 },
          attentionValueActivation: { $0 },
          intermediateSize: intermediateSize,
          intermediateActivation: intermediateActivation,
          hiddenDropoutProbability: hiddenDropoutProbability,
          attentionDropoutProbability: attentionDropoutProbability)
      }
    case let .albert(_, hiddenGroupCount):
      self.encoderLayers = (0..<hiddenGroupCount).map { _ in
        TransformerEncoderLayer(
          hiddenSize: hiddenSize,
          attentionHeadCount: attentionHeadCount,
          attentionQueryActivation: { $0 },
          attentionKeyActivation: { $0 },
          attentionValueActivation: { $0 },
          intermediateSize: intermediateSize,
          intermediateActivation: intermediateActivation,
          hiddenDropoutProbability: hiddenDropoutProbability,
          attentionDropoutProbability: attentionDropoutProbability)
      }
    }
  }

  /// Preprocesses an array of text sequences and prepares them for processing with BERT.
  /// Preprocessing mainly consists of tokenization.
  ///
  /// - Parameters:
  ///   - sequences: Text sequences (not tokenized).
  ///   - maxSequenceLength: Maximum sequence length supported by the text perception module. This
  ///     is mainly used for padding the preprocessed sequences. If not provided, it defaults to
  ///     this model's maximum supported sequence length.
  ///   - tokenizer: Tokenizer to use while preprocessing.
  ///
  /// - Returns: Text batch that can be processed by BERT.
  public func preprocess(sequences: [String], maxSequenceLength: Int? = nil) -> TextBatch {
    let maxSequenceLength = maxSequenceLength ?? self.maxSequenceLength
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
    var tokenTypeIds = [Int32(0)]
    for (sequenceId, sequence) in sequences.enumerated() {
      for token in sequence {
        tokens.append(token)
        tokenTypeIds.append(Int32(sequenceId))
      }
      tokens.append("[SEP]")
      tokenTypeIds.append(Int32(sequenceId))
    }
    let tokenIds = tokens.map { Int32(vocabulary.tokensToIds[$0]!) }

    // The mask is set to `true` for real tokens and `false` for padding tokens. This is so that
    // only real tokens are attended to.
    let mask = [Int32](repeating: 1, count: tokenIds.count)

    return TextBatch(
      tokenIds: Tensor(tokenIds).expandingShape(at: 0),
      tokenTypeIds: Tensor(tokenTypeIds).expandingShape(at: 0),
      mask: Tensor(mask).expandingShape(at: 0))
  }

  @differentiable(wrt: self)
  public func callAsFunction(_ input: TextBatch) -> Tensor<Scalar> {
    let sequenceLength = input.tokenIds.shape[1]

    // Compute the input embeddings and apply layer normalization and dropout on them.
    let tokenEmbeddings = tokenEmbedding(input.tokenIds)
    let tokenTypeEmbeddings = tokenTypeEmbedding(input.tokenTypeIds)
    let positionEmbeddings = positionEmbedding.embeddings.slice(
      lowerBounds: [0, 0],
      upperBounds: [sequenceLength, -1]
    ).expandingShape(at: 0)
    var embeddings = tokenEmbeddings + tokenTypeEmbeddings + positionEmbeddings
    embeddings = embeddingLayerNormalization(embeddings)
    embeddings = embeddingDropout(embeddings)

    if case .albert = variant {
      embeddings = embeddingProjection[0](embeddings)
    }

    // Create an attention mask for the inputs with shape
    // `[batchSize, sequenceLength, sequenceLength]`.
    let attentionMask = createAttentionMask(forTextBatch: input)

    // We keep the representation as a 2-D tensor to avoid reshaping it back and forth from a 3-D
    // tensor to a 2-D tensor. Reshapes are normally free on GPUs/CPUs but may not be free on TPUs,
    // and so we want to minimize them to help the optimizer.
    var transformerInput = embeddings.reshapedToMatrix()
    let batchSize = embeddings.shape[0]

    // Run the stacked transformer.
    switch variant {
    case .originalBert:
      for layerIndex in 0..<withoutDerivative(at: encoderLayers) { $0.count } {
        transformerInput = encoderLayers[layerIndex](TransformerInput(
          sequence: transformerInput,
          attentionMask: attentionMask,
          batchSize: batchSize))
      }
    case let .albert(_, hiddenGroupCount):
      let groupsPerLayer = Float(hiddenGroupCount) / Float(hiddenLayerCount)
      for layerIndex in 0..<hiddenLayerCount {
        let groupIndex = Int(Float(layerIndex) * groupsPerLayer)
        transformerInput = encoderLayers[groupIndex](TransformerInput(
          sequence: transformerInput,
          attentionMask: attentionMask,
          batchSize: batchSize))
      }
    }

    // Reshape back to the original tensor shape.
    return transformerInput.reshapedFromMatrix(originalShape: embeddings.shape)
  }
}

extension BERT {
  public enum Variant: CustomStringConvertible {
    /// - Source: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](
    ///            https://arxiv.org/pdf/1810.04805.pdf).
    case originalBert

    /// - Source: [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations](
    ///            https://arxiv.org/pdf/1909.11942.pdf).
    case albert(embeddingSize: Int, hiddenGroupCount: Int)

    public var description: String {
      switch self {
      case .originalBert:
        return "original-bert"
      case let .albert(embeddingSize, hiddenGroupCount):
        return "albert-E-\(embeddingSize)-G-\(hiddenGroupCount)"
      }
    }
  }
}

//===-----------------------------------------------------------------------------------------===//
// Pre-Trained Models
//===-----------------------------------------------------------------------------------------===//

extension BERT {
  public enum PreTrainedModel {
    case bertBase(cased: Bool, multilingual: Bool)
    case bertLarge(cased: Bool, wholeWordMasking: Bool)
    case albertBase
    case albertLarge
    case albertXLarge
    case albertXXLarge

    /// The name of this pre-trained model.
    public var name: String {
      switch self {
      case .bertBase(false, false): return "uncased_L-12_H-768_A-12"
      case .bertBase(true, false): return "cased_L-12_H-768_A-12"
      case .bertBase(false, true): return "multilingual_L-12_H-768_A-12"
      case .bertBase(true, true): return "multi_cased_L-12_H-768_A-12"
      case .bertLarge(false, false): return "uncased_L-24_H-1024_A-16"
      case .bertLarge(true, false): return "cased_L-24_H-1024_A-16"
      case .bertLarge(false, true): return "wwm_uncased_L-24_H-1024_A-16"
      case .bertLarge(true, true): return "wwm_cased_L-24_H-1024_A-16"
      case .albertBase: return "base"
      case .albertLarge: return "large"
      case .albertXLarge: return "xLarge"
      case .albertXXLarge: return "xxLarge"
      }
    }

    /// The URL where this pre-trained model can be downloaded from.
    public var url: URL {
      let bertPrefix = "https://storage.googleapis.com/bert_models"
      let albertPrefix = "https://storage.googleapis.com/tfhub-modules/google/albert"
      switch self {
      case .bertBase(false, false): return URL(string: "\(bertPrefix)/2018_10_18/\(name).zip")!
      case .bertBase(true, false): return URL(string: "\(bertPrefix)/2018_10_18/\(name).zip")!
      case .bertBase(false, true): return URL(string: "\(bertPrefix)/2018_11_03/\(name).zip")!
      case .bertBase(true, true): return URL(string: "\(bertPrefix)/2018_11_23/\(name).zip")!
      case .bertLarge(false, false): return URL(string: "\(bertPrefix)/2018_10_18/\(name).zip")!
      case .bertLarge(true, false): return URL(string: "\(bertPrefix)/2018_10_18/\(name).zip")!
      case .bertLarge(false, true): return URL(string: "\(bertPrefix)/2019_05_30/\(name).zip")!
      case .bertLarge(true, true): return URL(string: "\(bertPrefix)/2019_05_30/\(name).zip")!
      case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
        return URL(string: "\(albertPrefix)_\(name)/1.tar.gz")!
      }
    }

    public var variant: Variant {
      switch self {
      case .bertBase: return .originalBert
      case .bertLarge: return .originalBert
      case .albertBase: return .albert(embeddingSize: 128, hiddenGroupCount: 1)
      case .albertLarge: return .albert(embeddingSize: 128, hiddenGroupCount: 1)
      case .albertXLarge: return .albert(embeddingSize: 128, hiddenGroupCount: 1)
      case .albertXXLarge: return .albert(embeddingSize: 128, hiddenGroupCount: 1)
      }
    }

    public var caseSensitive: Bool {
      switch self {
      case let .bertBase(cased, _): return cased
      case let .bertLarge(cased, _): return cased
      case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge: return false
      }
    }

    public var hiddenSize: Int {
      switch self {
      case .bertBase: return 768
      case .bertLarge: return 1024
      case .albertBase: return 768
      case .albertLarge: return 1024
      case .albertXLarge: return 2048
      case .albertXXLarge: return 4096
      }
    }

    public var hiddenLayerCount: Int {
      switch self {
      case .bertBase: return 12
      case .bertLarge: return 24
      case .albertBase: return 12
      case .albertLarge: return 24
      case .albertXLarge: return 24
      case .albertXXLarge: return 12
      }
    }

    public var attentionHeadCount: Int {
      switch self {
      case .bertBase: return 12
      case .bertLarge: return 16
      case .albertBase: return 12
      case .albertLarge: return 16
      case .albertXLarge: return 16
      case .albertXXLarge: return 64
      }
    }

    public var intermediateSize: Int {
      switch self {
      case .bertBase: return 3072
      case .bertLarge: return 4096
      case .albertBase: return 3072
      case .albertLarge: return 4096
      case .albertXLarge: return 8192
      case .albertXXLarge: return 16384
      }
    }

    /// Loads this pre-trained BERT model from the specified directory.
    ///
    /// - Note: This function will download the pre-trained model files to the specified directory,
    ///   if they are not already there.
    ///
    /// - Parameters:
    ///   - directory: Directory to load the pretrained model from.
    public func load(from directory: URL) throws -> BERT {
      logger.info("Loading BERT pre-trained model '\(name)'.")
      let directory = directory.appendingPathComponent(variant.description)
      try maybeDownload(to: directory)

      // Load the appropriate vocabulary file.
      let vocabulary: Vocabulary = {
        switch self {
        case .bertBase, .bertLarge:
          let vocabularyURL = directory
            .appendingPathComponent(name)
            .appendingPathComponent("vocab.txt")
          return try! Vocabulary(fromFile: vocabularyURL)
        case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
          let vocabularyURL = directory
            .appendingPathComponent(name)
            .appendingPathComponent("assets")
            .appendingPathComponent("30k-clean.model")
          return try! Vocabulary(fromSentencePieceModel: vocabularyURL)
        }
      }()

      // Create a BERT model.
      var model = BERT(
        variant: variant,
        vocabulary: vocabulary,
        caseSensitive: caseSensitive,
        hiddenSize: hiddenSize,
        hiddenLayerCount: hiddenLayerCount,
        attentionHeadCount: attentionHeadCount,
        intermediateSize: intermediateSize,
        intermediateActivation: gelu,
        hiddenDropoutProbability: 0.1,
        attentionDropoutProbability: 0.1,
        maxSequenceLength: 512,
        typeVocabularySize: 2,
        initializerStandardDeviation: 0.02,
        useOneHotEmbeddings: false)

      // Load the pre-trained model checkpoint.
      switch self {
      case .bertBase, .bertLarge:
        model.load(fromTensorFlowCheckpoint: directory
          .appendingPathComponent(name)
          .appendingPathComponent("bert_model.ckpt"))
      case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
        model.load(fromTensorFlowCheckpoint: directory
          .appendingPathComponent(name)
          .appendingPathComponent("variables")
          .appendingPathComponent("variables"))
      }
      return model
    }

    /// Downloads this pre-trained model to the specified directory, if it's not already there.
    public func maybeDownload(to directory: URL) throws {
      switch self {
      case .bertBase, .bertLarge:
        // Download the model, if necessary.
        let compressedFileURL = directory.appendingPathComponent("\(name).zip")
        try NCA.maybeDownload(from: url, to: compressedFileURL)

        // Extract the data, if necessary.
        let extractedDirectoryURL = compressedFileURL.deletingPathExtension()
        if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
          try extract(zipFileAt: compressedFileURL, to: directory)
        }
      case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
        // Download the model, if necessary.
        let compressedFileURL = directory.appendingPathComponent("\(name).tar.gz")
        try NCA.maybeDownload(from: url, to: compressedFileURL)

        // Extract the data, if necessary.
        let extractedDirectoryURL = directory.appendingPathComponent(name)
        if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
          try extract(tarGZippedFileAt: compressedFileURL, to: extractedDirectoryURL)
        }
      }
    }
  }

  /// Loads a BERT model from the provided TensorFlow checkpoint file into this BERT model.
  ///
  /// - Parameters:
  ///   - fileURL: Path to the checkpoint file. Note that TensorFlow checkpoints typically consist
  ///     of multiple files (e.g., `bert_model.ckpt.index`, `bert_model.ckpt.meta`, and
  ///     `bert_model.ckpt.data-00000-of-00001`). In this case, the file URL should be specified as
  ///     their common prefix (e.g., `bert_model.ckpt`).
  public mutating func load(fromTensorFlowCheckpoint fileURL: URL) {
    let checkpointReader = TensorFlowCheckpointReader(checkpointPath: fileURL.path)
    tokenEmbedding.embeddings =
      Tensor(checkpointReader.loadTensor(named: "bert/embeddings/word_embeddings"))
    tokenTypeEmbedding.embeddings =
      Tensor(checkpointReader.loadTensor(named: "bert/embeddings/token_type_embeddings"))
    positionEmbedding.embeddings =
      Tensor(checkpointReader.loadTensor(named: "bert/embeddings/position_embeddings"))
    embeddingLayerNormalization.offset =
      Tensor(checkpointReader.loadTensor(named: "bert/embeddings/LayerNorm/beta"))
    embeddingLayerNormalization.scale =
      Tensor(checkpointReader.loadTensor(named: "bert/embeddings/LayerNorm/gamma"))
    switch variant {
    case .originalBert:
      for layerIndex in encoderLayers.indices {
        let prefix = "bert/encoder/layer_\(layerIndex)"
        encoderLayers[layerIndex].multiHeadAttention.queryWeight =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/self/query/kernel"))
        encoderLayers[layerIndex].multiHeadAttention.queryBias =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/self/query/bias"))
        encoderLayers[layerIndex].multiHeadAttention.keyWeight =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/self/key/kernel"))
        encoderLayers[layerIndex].multiHeadAttention.keyBias =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/self/key/bias"))
        encoderLayers[layerIndex].multiHeadAttention.valueWeight =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/self/value/kernel"))
        encoderLayers[layerIndex].multiHeadAttention.valueBias =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/self/value/bias"))
        encoderLayers[layerIndex].attentionWeight =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/output/dense/kernel"))
        encoderLayers[layerIndex].attentionBias =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/output/dense/bias"))
        encoderLayers[layerIndex].attentionLayerNormalization.offset =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/output/LayerNorm/beta"))
        encoderLayers[layerIndex].attentionLayerNormalization.scale =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention/output/LayerNorm/gamma"))
        encoderLayers[layerIndex].intermediateWeight =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/intermediate/dense/kernel"))
        encoderLayers[layerIndex].intermediateBias =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/intermediate/dense/bias"))
        encoderLayers[layerIndex].outputWeight =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/output/dense/kernel"))
        encoderLayers[layerIndex].outputBias =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/output/dense/bias"))
        encoderLayers[layerIndex].outputLayerNormalization.offset =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/output/LayerNorm/beta"))
        encoderLayers[layerIndex].outputLayerNormalization.scale =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/output/LayerNorm/gamma"))
      }
    case .albert:
      embeddingProjection[0].weight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/embedding_hidden_mapping_in/kernel"))
      embeddingProjection[0].bias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/embedding_hidden_mapping_in/bias"))
      for layerIndex in encoderLayers.indices {
        let prefix = "bert/encoder/transformer/group_\(layerIndex)/inner_group_0"
        encoderLayers[layerIndex].multiHeadAttention.queryWeight =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/self/query/kernel"))
        encoderLayers[layerIndex].multiHeadAttention.queryBias =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/self/query/bias"))
        encoderLayers[layerIndex].multiHeadAttention.keyWeight =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/self/key/kernel"))
        encoderLayers[layerIndex].multiHeadAttention.keyBias =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/self/key/bias"))
        encoderLayers[layerIndex].multiHeadAttention.valueWeight =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/self/value/kernel"))
        encoderLayers[layerIndex].multiHeadAttention.valueBias =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/self/value/bias"))
        encoderLayers[layerIndex].attentionWeight =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/output/dense/kernel"))
        encoderLayers[layerIndex].attentionBias =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/attention_1/output/dense/bias"))
        encoderLayers[layerIndex].attentionLayerNormalization.offset =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/LayerNorm/beta"))
        encoderLayers[layerIndex].attentionLayerNormalization.scale =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/LayerNorm/gamma"))
        encoderLayers[layerIndex].intermediateWeight =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/ffn_1/intermediate/dense/kernel"))
        encoderLayers[layerIndex].intermediateBias =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/ffn_1/intermediate/dense/bias"))
        encoderLayers[layerIndex].outputWeight =
          Tensor(checkpointReader.loadTensor(
            named: "\(prefix)/ffn_1/intermediate/output/dense/kernel"))
        encoderLayers[layerIndex].outputBias =
          Tensor(checkpointReader.loadTensor(
            named: "\(prefix)/ffn_1/intermediate/output/dense/bias"))
        encoderLayers[layerIndex].outputLayerNormalization.offset =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/LayerNorm_1/beta"))
        encoderLayers[layerIndex].outputLayerNormalization.scale =
          Tensor(checkpointReader.loadTensor(named: "\(prefix)/LayerNorm_1/gamma"))
      }
    }
  }
}
