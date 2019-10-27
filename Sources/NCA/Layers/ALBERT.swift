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

/// ALBERT layer for encoding text.
///
/// - Source: [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations](
///             https://arxiv.org/pdf/1909.11942.pdf).
public struct ALBERT: Module, Regularizable { // <Scalar: TensorFlowFloatingPoint & Codable>: Module {
  // TODO: !!! Convert to a generic constraint once TF-427 is resolved.
  public typealias Scalar = Float

  @noDerivative public let configuration: Configuration
  public var tokenEmbedding: Embedding<Scalar>
  public var tokenTypeEmbedding: Embedding<Scalar>
  public var positionEmbedding: Embedding<Scalar>
  public var embeddingLayerNormalization: LayerNormalization<Scalar>
  public var embeddingDropout: Dropout<Scalar>
  public var embeddingProjection: Affine<Scalar>
  public var transformerEncoderLayers: [TransformerEncoderLayer]

  public var regularizationValue: TangentVector {
    TangentVector(
      tokenEmbedding: tokenEmbedding.regularizationValue,
      tokenTypeEmbedding: tokenTypeEmbedding.regularizationValue,
      positionEmbedding: positionEmbedding.regularizationValue,
      embeddingLayerNormalization: embeddingLayerNormalization.regularizationValue,
      embeddingDropout: embeddingDropout.regularizationValue,
      embeddingProjection: embeddingProjection.regularizationValue,
      transformerEncoderLayers: [TransformerEncoderLayer].TangentVector(
        transformerEncoderLayers.map { $0.regularizationValue }))
  }

  public init(configuration: Configuration, useOneHotEmbeddings: Bool = false) {
    self.configuration = configuration

    self.tokenEmbedding = Embedding<Scalar>(
      vocabularySize: configuration.vocabularySize,
      embeddingSize: configuration.embeddingSize,
      embeddingsInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor<Scalar>(configuration.initializerStandardDeviation)),
      useOneHotEmbeddings: useOneHotEmbeddings)

    // The token type vocabulary will always be small and so we use the one-hot approach here as
    // it is always faster for small vocabularies.
    self.tokenTypeEmbedding = Embedding<Scalar>(
      vocabularySize: configuration.typeVocabularySize,
      embeddingSize: configuration.embeddingSize,
      embeddingsInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor<Scalar>(configuration.initializerStandardDeviation)),
      useOneHotEmbeddings: true)

    // Since the position embeddings table is a learned variable, we create it using a (long)
    // sequence length, `maxSequenceLength`. The actual sequence length might be shorter than this,
    // for faster training of tasks that do not have long sequences. So, `positionEmbedding`
    // effectively contains an embedding table for positions
    // [0, 1, 2, ..., maxPositionEmbeddings - 1], and the current sequence may have positions
    // [0, 1, 2, ..., sequenceLength - 1], so we can just perform a slice.
    self.positionEmbedding = Embedding(
      vocabularySize: configuration.maxSequenceLength,
      embeddingSize: configuration.embeddingSize,
      embeddingsInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(configuration.initializerStandardDeviation)),
      useOneHotEmbeddings: false)

    self.embeddingLayerNormalization = LayerNormalization<Scalar>(
      featureCount: configuration.hiddenSize,
      axis: -1)
    self.embeddingDropout = Dropout(probability: configuration.hiddenDropoutProbability)
    self.embeddingProjection = Affine<Scalar>(
      inputSize: configuration.embeddingSize,
      outputSize: configuration.hiddenSize,
      weightInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(configuration.initializerStandardDeviation)))

    self.transformerEncoderLayers = (0..<configuration.hiddenGroupCount).map { _ in
      TransformerEncoderLayer(
        hiddenSize: configuration.hiddenSize,
        attentionHeadCount: configuration.attentionHeadCount,
        attentionQueryActivation: { x in x },
        attentionKeyActivation: { x in x },
        attentionValueActivation: { x in x },
        intermediateSize: configuration.intermediateSize,
        intermediateActivation: configuration.intermediateActivation.activationFunction(),
        hiddenDropoutProbability: configuration.hiddenDropoutProbability,
        attentionDropoutProbability: configuration.attentionDropoutProbability)
    }
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
    embeddings = embeddingProjection(embeddings)

    // Create an attention mask for the inputs with shape
    // `[batchSize, sequenceLength, sequenceLength]`.
    let attentionMask = createAttentionMask(forTextBatch: input)

    // We keep the representation as a 2-D tensor to avoid reshaping it back and forth from a 3-D
    // tensor to a 2-D tensor. Reshapes are normally free on GPUs/CPUs but may not be free on TPUs,
    // and so we want to minimize them to help the optimizer.
    var transformerInput = embeddings.reshapedToMatrix()
    let batchSize = embeddings.shape[0]
    let hiddenGroupCount = configuration.hiddenGroupCount
    let hiddenLayerCount = configuration.hiddenLayerCount
    let groupsPerLayer = Float(hiddenGroupCount) / Float(hiddenLayerCount)

    // Run the stacked transformer with the grouped layers.
    for layerIndex in 0..<hiddenLayerCount {
      let groupIndex = Int(Float(layerIndex) * groupsPerLayer)
      transformerInput = transformerEncoderLayers[groupIndex](TransformerInput(
        sequence: transformerInput,
        attentionMask: attentionMask,
        batchSize: batchSize))
    }

    return transformerInput.reshapedFromMatrix(originalShape: embeddings.shape)
  }
}

extension ALBERT {
  /// ALBERT configuration.
  public struct Configuration: Codable {
    public let vocabularySize: Int
    public let embeddingSize: Int
    public let hiddenSize: Int
    public let hiddenLayerCount: Int
    public let hiddenGroupCount: Int
    public let attentionHeadCount: Int
    public let intermediateSize: Int
    public let intermediateActivation: Activation
    public let hiddenDropoutProbability: Scalar
    public let attentionDropoutProbability: Scalar
    public let maxSequenceLength: Int
    public let typeVocabularySize: Int
    public let initializerStandardDeviation: Scalar

    /// Creates a new ALBERT configuration.
    ///
    /// - Parameters:
    ///   - vocabularySize: Vocabulary size.
    ///   - embeddingSize: Size of the vocabulary embeddings.
    ///   - hiddenSize: Size of the encoder and the pooling layers.
    ///   - hiddenLayerCount: Number of hidden layers in the encoder.
    ///   - hiddenGroupCount: Number of hidden layer groups in the encoder (that share parameters).
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
      vocabularySize: Int,
      embeddingSize: Int = 128,
      hiddenSize: Int = 4096,
      hiddenLayerCount: Int = 12,
      hiddenGroupCount: Int = 1,
      attentionHeadCount: Int = 64,
      intermediateSize: Int = 16384,
      intermediateActivation: Activation = .gelu,
      hiddenDropoutProbability: Scalar = 0.1,
      attentionDropoutProbability: Scalar = 0.1,
      maxSequenceLength: Int = 512,
      typeVocabularySize: Int = 16,
      initializerStandardDeviation: Scalar = 0.02
    ) {
      precondition(
        hiddenGroupCount <= hiddenLayerCount,
        "The number of hidden groups must not be greater than the number of hidden layers.")
      self.vocabularySize = vocabularySize
      self.embeddingSize = embeddingSize
      self.hiddenSize = hiddenSize
      self.hiddenLayerCount = hiddenLayerCount
      self.hiddenGroupCount = hiddenGroupCount
      self.attentionHeadCount = attentionHeadCount
      self.intermediateSize = intermediateSize
      self.intermediateActivation = intermediateActivation
      self.hiddenDropoutProbability = hiddenDropoutProbability
      self.attentionDropoutProbability = attentionDropoutProbability
      self.maxSequenceLength = maxSequenceLength
      self.typeVocabularySize = typeVocabularySize
      self.initializerStandardDeviation = initializerStandardDeviation
    }

    enum CodingKeys: String, CodingKey {
      case vocabularySize = "vocab_size"
      case embeddingSize = "embedding_size"
      case hiddenSize = "hidden_size"
      case hiddenLayerCount = "num_hidden_layers"
      case hiddenGroupCount = "num_hidden_groups"
      case attentionHeadCount = "num_attention_heads"
      case intermediateSize = "intermediate_size"
      case intermediateActivation = "hidden_act"
      case hiddenDropoutProbability = "hidden_dropout_prob"
      case attentionDropoutProbability = "attention_probs_dropout_prob"
      case maxSequenceLength = "max_position_embeddings"
      case typeVocabularySize = "type_vocab_size"
      case initializerStandardDeviation = "initializer_range"
    }
  }
}

extension ALBERT.Configuration {
  public init(fromJson json: String) throws {
    self = try JSONDecoder().decode(ALBERT.Configuration.self, from: json.data(using: .utf8)!)
  }

  public func toJson(pretty: Bool = true) throws -> String {
    let encoder = JSONEncoder()
    if pretty {
      encoder.outputFormatting = .prettyPrinted
    }
    let data = try encoder.encode(self)
    return String(data: data, encoding: .utf8)!
  }
}

extension ALBERT.Configuration: Serializable {
  public init(fromFile fileURL: URL) throws {
    try self.init(fromJson: try String(contentsOfFile: fileURL.path, encoding: .utf8))
  }

  public func save(toFile fileURL: URL) throws {
    try toJson().write(to: fileURL, atomically: true, encoding: .utf8)
  }
}

extension ALBERT {
  public enum Activation: String, Codable {
    case linear, relu, gelu, tanh

    public func activationFunction<Scalar: TensorFlowFloatingPoint>() -> NCA.Activation<Scalar> {
      switch self {
      case .linear: return { $0 }
      case .relu: return TensorFlow.relu
      case .gelu: return TensorFlow.gelu
      case .tanh: return TensorFlow.tanh
      }
    }
  }
}

//===-----------------------------------------------------------------------------------------===//
// Pre-Trained Models
//===-----------------------------------------------------------------------------------------===//

extension ALBERT {
  public enum PreTrainedModel {
    case base
    case large
    case xLarge
    case xxLarge

    /// The name of this pre-trained model.
    public var name: String {
      switch self {
      case .base: return "base"
      case .large: return "large"
      case .xLarge: return "xLarge"
      case .xxLarge: return "xxLarge"
      }
    }

    /// The configuration of this pre-trained model.
    public var configuration: Configuration {
      switch self {
      case .base: return Configuration(
        vocabularySize: 30000,
        embeddingSize: 128,
        hiddenSize: 768,
        hiddenLayerCount: 12,
        hiddenGroupCount: 1,
        attentionHeadCount: 12,
        intermediateSize: 3072,
        intermediateActivation: .gelu,
        hiddenDropoutProbability: 0.1,
        attentionDropoutProbability: 0.1,
        maxSequenceLength: 512,
        typeVocabularySize: 2,
        initializerStandardDeviation: 0.02)
      case .large: return Configuration(
        vocabularySize: 30000,
        embeddingSize: 128,
        hiddenSize: 1024,
        hiddenLayerCount: 24,
        hiddenGroupCount: 1,
        attentionHeadCount: 16,
        intermediateSize: 4096,
        intermediateActivation: .gelu,
        hiddenDropoutProbability: 0.1,
        attentionDropoutProbability: 0.1,
        maxSequenceLength: 512,
        typeVocabularySize: 2,
        initializerStandardDeviation: 0.02)
      case .xLarge: return Configuration(
        vocabularySize: 30000,
        embeddingSize: 128,
        hiddenSize: 2048,
        hiddenLayerCount: 24,
        hiddenGroupCount: 1,
        attentionHeadCount: 16,
        intermediateSize: 8192,
        intermediateActivation: .gelu,
        hiddenDropoutProbability: 0.1,
        attentionDropoutProbability: 0.1,
        maxSequenceLength: 512,
        typeVocabularySize: 2,
        initializerStandardDeviation: 0.02)
      case .xxLarge: return Configuration(
        vocabularySize: 30000,
        embeddingSize: 128,
        hiddenSize: 4096,
        hiddenLayerCount: 12,
        hiddenGroupCount: 1,
        attentionHeadCount: 64,
        intermediateSize: 16384,
        intermediateActivation: .gelu,
        hiddenDropoutProbability: 0.1,
        attentionDropoutProbability: 0.1,
        maxSequenceLength: 512,
        typeVocabularySize: 2,
        initializerStandardDeviation: 0.02)
      }
    }

    /// The URL where this pre-trained model can be downloaded from.
    public var url: URL {
      URL(string: "https://storage.googleapis.com/tfhub-modules/google/albert_\(name)/1.tar.gz")!
    }

    /// Downloads this pre-trained model to the specified directory, if it's not already there.
    public func maybeDownload(to directory: URL) throws {
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

  /// Loads a pre-trained ALBERT model from the specified directory.
  ///
  /// - Note: This function will download the pre-trained model files to the specified directory,
  ///   if they are not already there.
  ///
  /// - Parameters:
  ///   - model: Pre-trained model configuration to load.
  ///   - directory: Directory to load the pretrained model from.
  public mutating func load(preTrainedModel model: PreTrainedModel, from directory: URL) throws {
    logger.info("Loading ALBERT pre-trained model '\(model.name)'.")

    // Download the model, if necessary.
    try model.maybeDownload(to: directory)

    // Load the model.
    load(fromTensorFlowCheckpoint: directory
      .appendingPathComponent("variables")
      .appendingPathComponent("variables"))
  }

  /// Loads a ALBERT model from the provided TensorFlow checkpoint file into this ALBERT model.
  ///
  /// - Parameters:
  ///   - fileURL: Path to the checkpoint file. Note that TensorFlow checkpoints typically consist
  ///     of multiple files (e.g., `variables.index` `variables.data-00000-of-00001`). In this
  ///     case, the file URL should be specified as their common prefix (e.g., `variables`).
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
    embeddingProjection.weight =
      Tensor(checkpointReader.loadTensor(named: "bert/encoder/embedding_hidden_mapping_in/kernel"))
    embeddingProjection.bias =
      Tensor(checkpointReader.loadTensor(named: "bert/encoder/embedding_hidden_mapping_in/bias"))
    for layerIndex in transformerEncoderLayers.indices {
      transformerEncoderLayers[layerIndex].multiHeadAttention.queryWeight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/inner_group_0/attention_1/self/query/kernel"))
      transformerEncoderLayers[layerIndex].multiHeadAttention.queryBias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/inner_group_0/attention_1/self/query/bias"))
      transformerEncoderLayers[layerIndex].multiHeadAttention.keyWeight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/inner_group_0/attention_1/self/key/kernel"))
      transformerEncoderLayers[layerIndex].multiHeadAttention.keyBias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/inner_group_0/attention_1/self/key/bias"))
      transformerEncoderLayers[layerIndex].multiHeadAttention.valueWeight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/inner_group_0/attention_1/self/value/kernel"))
      transformerEncoderLayers[layerIndex].multiHeadAttention.valueBias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/inner_group_0/attention_1/self/value/bias"))
      transformerEncoderLayers[layerIndex].attentionWeight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/inner_group_0/attention_1/output/dense/kernel"))
      transformerEncoderLayers[layerIndex].attentionBias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/inner_group_0/attention_1/output/dense/bias"))
      transformerEncoderLayers[layerIndex].attentionLayerNormalization.offset =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/inner_group_0/LayerNorm/beta"))
      transformerEncoderLayers[layerIndex].attentionLayerNormalization.scale =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/inner_group_0/LayerNorm/gamma"))
      transformerEncoderLayers[layerIndex].intermediateWeight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/ffn_1/intermediate/dense/kernel"))
      transformerEncoderLayers[layerIndex].intermediateBias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/ffn_1/intermediate/dense/bias"))
      transformerEncoderLayers[layerIndex].outputWeight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/ffn_1/intermediate/output/dense/kernel"))
      transformerEncoderLayers[layerIndex].outputBias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/ffn_1/intermediate/output/dense/bias"))
      transformerEncoderLayers[layerIndex].outputLayerNormalization.offset =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/inner_group_0/LayerNorm_1/beta"))
      transformerEncoderLayers[layerIndex].outputLayerNormalization.scale =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/transformer/group_\(layerIndex)/inner_group_0/LayerNorm_1/gamma"))
    }
  }
}
