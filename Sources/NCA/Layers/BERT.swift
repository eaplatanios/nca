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
/// - Source: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](
///             https://arxiv.org/pdf/1810.04805.pdf).
public struct BERT: Module { // <Scalar: TensorFlowFloatingPoint & Codable>: Module {
  // TODO: !!! Convert to a generic constraint once TF-427 is resolved.
  public typealias Scalar = Float

  @noDerivative public let configuration: Configuration
  public var tokenEmbedding: Embedding<Scalar>
  public var tokenTypeEmbedding: Embedding<Scalar>
  public var positionEmbedding: Embedding<Scalar>
  public var embeddingLayerNormalization: LayerNormalization<Scalar>
  public var embeddingDropout: Dropout<Scalar>
  public var transformerEncoder: TransformerEncoder // TODO: !!! <Scalar>

  public init(configuration: Configuration, useOneHotEmbeddings: Bool = false) {
    self.configuration = configuration

    self.tokenEmbedding = Embedding<Scalar>(
      vocabularySize: configuration.vocabularySize,
      embeddingSize: configuration.hiddenSize,
      embeddingsInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor<Scalar>(configuration.initializerStandardDeviation)),
      useOneHotEmbeddings: useOneHotEmbeddings)

    // The token type vocabulary will always be small and so we use the one-hot approach here as
    // it is always faster for small vocabularies.
    self.tokenTypeEmbedding = Embedding<Scalar>(
      vocabularySize: configuration.typeVocabularySize,
      embeddingSize: configuration.hiddenSize,
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
      embeddingSize: configuration.hiddenSize,
      embeddingsInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(configuration.initializerStandardDeviation)),
      useOneHotEmbeddings: false)

    self.embeddingLayerNormalization = LayerNormalization<Scalar>(
      featureCount: configuration.hiddenSize,
      axis: -1)
    self.embeddingDropout = Dropout(probability: configuration.hiddenDropoutProbability)

    self.transformerEncoder = TransformerEncoder(
      hiddenSize: configuration.hiddenSize,
      layerCount: configuration.hiddenLayerCount,
      attentionHeadCount: configuration.attentionHeadCount,
      attentionQueryActivation: { x in x },
      attentionKeyActivation: { x in x },
      attentionValueActivation: { x in x },
      intermediateSize: configuration.intermediateSize,
      intermediateActivation: configuration.intermediateActivation.activationFunction(),
      hiddenDropoutProbability: configuration.hiddenDropoutProbability,
      attentionDropoutProbability: configuration.attentionDropoutProbability)
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

    // Create an attention mask for the inputs with shape
    // `[batchSize, sequenceLength, sequenceLength]`.
    let attentionMask = createAttentionMask(forInput: input)

    // Run the stacked transformer.
    return transformerEncoder(TransformerInput(
      sequence: embeddings,
      attentionMask: attentionMask))
  }
}

extension BERT {
  /// BERT configuration.
  public struct Configuration: Codable {
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let hiddenLayerCount: Int
    public let attentionHeadCount: Int
    public let intermediateSize: Int
    public let intermediateActivation: Activation
    public let hiddenDropoutProbability: Scalar
    public let attentionDropoutProbability: Scalar
    public let maxSequenceLength: Int
    public let typeVocabularySize: Int
    public let initializerStandardDeviation: Scalar

    /// Creates a new BERT configuration.
    ///
    /// - Parameters:
    ///   - vocabularySize: Vocabulary size.
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
      vocabularySize: Int,
      hiddenSize: Int = 768,
      hiddenLayerCount: Int = 12,
      attentionHeadCount: Int = 12,
      intermediateSize: Int = 3072,
      intermediateActivation: Activation = .gelu,
      hiddenDropoutProbability: Scalar = 0.1,
      attentionDropoutProbability: Scalar = 0.1,
      maxSequenceLength: Int = 512,
      typeVocabularySize: Int = 16,
      initializerStandardDeviation: Scalar = 0.02
    ) {
      self.vocabularySize = vocabularySize
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
    }

    enum CodingKeys: String, CodingKey {
      case vocabularySize = "vocab_size"
      case hiddenSize = "hidden_size"
      case hiddenLayerCount = "num_hidden_layers"
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

extension BERT.Configuration {
  public init(fromJson json: String) throws {
    self = try JSONDecoder().decode(BERT.Configuration.self, from: json.data(using: .utf8)!)
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

extension BERT.Configuration: Serializable {
  public init(fromFile fileURL: URL) throws {
    try self.init(fromJson: try String(contentsOfFile: fileURL.path, encoding: .utf8))
  }

  public func save(toFile fileURL: URL) throws {
    try toJson().write(to: fileURL, atomically: true, encoding: .utf8)
  }
}

extension BERT {
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

extension BERT {
  /// Returns a 3-D attention mask that correspond to the 2-D mask of the provided input.
  ///
  /// - Parameters:
  ///   - input: BERT input for which to create an attention mask. `input.mask` has shape
  ///     `[batchSize, sequenceLength]`.
  ///
  /// - Returns: Attention mask with shape `[batchSize, sequenceLength, sequenceLength]`.
  internal func createAttentionMask(forInput input: TextBatch) -> Tensor<Scalar> {
    let batchSize = input.tokenIds.shape[0]
    let fromSequenceLength = input.tokenIds.shape[1]
    let toSequenceLength = input.mask.shape[1]
    let reshapedMask = Tensor<Scalar>(input.mask.reshaped(to: [batchSize, 1, toSequenceLength]))

    // We do not assume that `input.tokenIds` is a mask. We do not actually care if we attend
    // *from* padding tokens (only *to* padding tokens) so we create a tensor of all ones.
    let broadcastOnes = Tensor<Scalar>(ones: [batchSize, fromSequenceLength, 1])

    // We broadcast along two dimensions to create the mask.
    return broadcastOnes * reshapedMask
  }
}

//===-----------------------------------------------------------------------------------------===//
// Pre-Trained Models
//===-----------------------------------------------------------------------------------------===//

extension BERT {
  public enum PreTrainedModel {
    case base(cased: Bool, multilingual: Bool)
    case large(cased: Bool, wholeWordMasking: Bool)

    /// The name of this pre-trained model.
    public var name: String {
      switch self {
      case .base(false, false): return "uncased_L-12_H-768_A-12"
      case .base(true, false): return "cased_L-12_H-768_A-12"
      case .base(false, true): return "multilingual_L-12_H-768_A-12"
      case .base(true, true): return "multi_cased_L-12_H-768_A-12"
      case .large(false, false): return "uncased_L-24_H-1024_A-16"
      case .large(true, false): return "cased_L-24_H-1024_A-16"
      case .large(false, true): return "wwm_uncased_L-24_H-1024_A-16"
      case .large(true, true): return "wwm_cased_L-24_H-1024_A-16"
      }
    }

    /// The URL where this pre-trained model can be downloaded from.
    public var url: URL {
      let prefix = "https://storage.googleapis.com/bert_models"
      switch self {
      case .base(false, false): return URL(string: "\(prefix)/2018_10_18/\(name).zip")!
      case .base(true, false): return URL(string: "\(prefix)/2018_10_18/\(name).zip")!
      case .base(false, true): return URL(string: "\(prefix)/2018_11_03/\(name).zip")!
      case .base(true, true): return URL(string: "\(prefix)/2018_11_23/\(name).zip")!
      case .large(false, false): return URL(string: "\(prefix)/2018_10_18/\(name).zip")!
      case .large(true, false): return URL(string: "\(prefix)/2018_10_18/\(name).zip")!
      case .large(false, true): return URL(string: "\(prefix)/2019_05_30/\(name).zip")!
      case .large(true, true): return URL(string: "\(prefix)/2019_05_30/\(name).zip")!
      }
    }

    /// Downloads this pre-trained model to the specified directory, if it's not already there.
    public func maybeDownload(to directory: URL) throws {
      // Download the model, if necessary.
      let compressedFileURL = directory.appendingPathComponent("\(name).zip")
      try NCA.maybeDownload(from: url, to: compressedFileURL)

      // Extract the data, if necessary.
      let extractedDirectoryURL = compressedFileURL.deletingPathExtension()
      if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
        try FileManager.default.unzipItem(at: compressedFileURL, to: directory)
      }
    }
  }

  /// Loads a pre-trained BERT model from the specified directory.
  ///
  /// - Note: This function will download the pre-trained model files to the specified directory,
  ///   if they are not already there.
  ///
  /// - Parameters:
  ///   - model: Pre-trained model configuration to load.
  ///   - directory: Directory to load the pretrained model from.
  public mutating func load(preTrainedModel model: PreTrainedModel, from directory: URL) throws {
    // Download the model, if necessary.
    try model.maybeDownload(to: directory)

    // Load the model.
    load(fromTensorFlowCheckpoint: directory
      .appendingPathComponent(model.name)
      .appendingPathComponent("bert_model.ckpt"))
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
    for layerIndex in transformerEncoder.encoderLayers.indices {
      transformerEncoder.encoderLayers[layerIndex].multiHeadAttention.queryWeight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/attention/self/query/kernel"))
      transformerEncoder.encoderLayers[layerIndex].multiHeadAttention.queryBias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/attention/self/query/bias"))
      transformerEncoder.encoderLayers[layerIndex].multiHeadAttention.keyWeight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/attention/self/key/kernel"))
      transformerEncoder.encoderLayers[layerIndex].multiHeadAttention.keyBias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/attention/self/key/bias"))
      transformerEncoder.encoderLayers[layerIndex].multiHeadAttention.valueWeight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/attention/self/value/kernel"))
      transformerEncoder.encoderLayers[layerIndex].multiHeadAttention.valueBias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/attention/self/value/bias"))
      transformerEncoder.encoderLayers[layerIndex].attentionWeight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/attention/output/dense/kernel"))
      transformerEncoder.encoderLayers[layerIndex].attentionBias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/attention/output/dense/bias"))
      transformerEncoder.encoderLayers[layerIndex].attentionLayerNormalization.offset =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/attention/output/LayerNorm/beta"))
      transformerEncoder.encoderLayers[layerIndex].attentionLayerNormalization.scale =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/attention/output/LayerNorm/gamma"))
      transformerEncoder.encoderLayers[layerIndex].intermediateWeight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/intermediate/dense/kernel"))
      transformerEncoder.encoderLayers[layerIndex].intermediateBias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/intermediate/dense/bias"))
      transformerEncoder.encoderLayers[layerIndex].outputWeight =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/output/dense/kernel"))
      transformerEncoder.encoderLayers[layerIndex].outputBias =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/output/dense/bias"))
      transformerEncoder.encoderLayers[layerIndex].outputLayerNormalization.offset =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/output/LayerNorm/beta"))
      transformerEncoder.encoderLayers[layerIndex].outputLayerNormalization.scale =
        Tensor(checkpointReader.loadTensor(
          named: "bert/encoder/layer_\(layerIndex)/output/LayerNorm/gamma"))
    }
  }
}
