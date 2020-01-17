// Copyright 2020, Emmanouil Antonios Platanios. All Rights Reserved.
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

import Core
import TensorFlow

public struct MLP: Layer {
  @noDerivative public let dropout: Dropout<Float> = Dropout<Float>(probability: 0.2)
  public var dense1 = Dense<Float>(inputSize: 28 * 28 * 3, outputSize: 1024, activation: gelu)
  public var dense2 = Dense<Float>(inputSize: 1024, outputSize: 512, activation: gelu)
  public var dense3 = Dense<Float>(inputSize: 512, outputSize: 256, activation: gelu)
  public var dense4 = Dense<Float>(inputSize: 256, outputSize: 10)

  public init() {}

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    dense4(
      dropout(dense3(
        dropout(dense2(
          dropout(dense1(input.reshaped(to: [-1, 28 * 28 * 3])))
        ))
      )))
  }
}

public struct ReverseContextualizedMLP: Layer {
  @noDerivative public let dropout: Dropout<Float> = Dropout<Float>(probability: 0.2)
  @noDerivative public let dense1Base: Dense<Float>
  public var dense1Generator: Sequential<Dense<Float>, Dense<Float>>
  @noDerivative public let dense2Base: Dense<Float>
  public var dense2Generator: Sequential<Dense<Float>, Dense<Float>>
  @noDerivative public let dense3Base: Dense<Float>
  public var dense3Generator: Sequential<Dense<Float>, Dense<Float>>
  @noDerivative public let dense4Base: Dense<Float>
  public var dense4Generator: Sequential<Dense<Float>, Dense<Float>>
  public var initialContext: Tensor<Float>

  public init(functionEmbeddingSize: Int) {
    let dense4Base = Dense<Float>(inputSize: 256, outputSize: 10)
    self.dense4Base = dense4Base
    self.dense4Generator = Sequential {
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: dense4Base.parameterCount)
    }
    let dense3Base = Dense<Float>(inputSize: 512, outputSize: 256, activation: gelu)
    self.dense3Base = dense3Base
    self.dense3Generator = Sequential {
      Dense<Float>(inputSize: dense4Base.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: dense3Base.parameterCount)
    }
    let dense2Base = Dense<Float>(inputSize: 1024, outputSize: 512, activation: gelu)
    self.dense2Base = dense2Base
    self.dense2Generator = Sequential {
      Dense<Float>(inputSize: dense3Base.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: dense2Base.parameterCount)
    }
    let dense1Base = Dense<Float>(inputSize: 28 * 28 * 3, outputSize: 1024, activation: gelu)
    self.dense1Base = dense1Base
    self.dense1Generator = Sequential {
      Dense<Float>(inputSize: dense2Base.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: dense1Base.parameterCount)
    }
    self.initialContext = Tensor<Float>(randomNormal: [1, functionEmbeddingSize])
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let dense4Parameters = dense4Generator(initialContext)
    let dense3Parameters = dense3Generator(dense4Parameters)
    let dense2Parameters = dense2Generator(dense3Parameters)
    let dense1Parameters = dense1Generator(dense2Parameters)
    let dense1 = Dense<Float>(unflattening: dense1Parameters, like: dense1Base)
    let dense2 = Dense<Float>(unflattening: dense2Parameters, like: dense2Base)
    let dense3 = Dense<Float>(unflattening: dense3Parameters, like: dense3Base)
    let dense4 = Dense<Float>(unflattening: dense4Parameters, like: dense4Base)
    return dense4(
      dropout(dense3(
        dropout(dense2(
          dropout(dense1(input.reshaped(to: [-1, 28 * 28 * 3])))
        ))
      )))
  }
}

public struct ReverseContextualizedLeNet: Layer {
  @noDerivative public let conv1Base: Conv2D<Float>
  public var conv1Generator: Sequential<Dense<Float>, Dense<Float>>
  public var pool1: MaxPool2D<Float>
  @noDerivative public let conv2Base: Conv2D<Float>
  public var conv2Generator: Sequential<Dense<Float>, Dense<Float>>
  public var pool2: MaxPool2D<Float>
  public var flatten: Flatten<Float>
  @noDerivative public let fc1Base: Dense<Float>
  public var fc1Generator: Sequential<Dense<Float>, Dense<Float>>
  public var fc2: Dense<Float>

  public init(functionEmbeddingSize: Int) {
    let fc2 = Dense<Float>(inputSize: 1024, outputSize: 10)
    let fc1Base = Dense<Float>(inputSize: 7 * 7 * 64, outputSize: 1024, activation: gelu)
    self.fc2 = fc2
    self.fc1Base = fc1Base
    self.fc1Generator = Sequential {
      Dense<Float>(inputSize: fc2.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: fc1Base.parameterCount)
    }
    self.flatten = Flatten<Float>()
    self.pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    let conv2Base = Conv2D<Float>(filterShape: (5, 5, 32, 64), padding: .same, activation: gelu)
    self.conv2Base = conv2Base
    self.conv2Generator = Sequential {
      Dense<Float>(inputSize: fc1Base.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: conv2Base.parameterCount)
    }
    self.pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    let conv1Base = Conv2D<Float>(filterShape: (5, 5, 3, 32), padding: .same, activation: gelu)
    self.conv1Base = conv1Base
    self.conv1Generator = Sequential {
      Dense<Float>(inputSize: conv2Base.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: conv1Base.parameterCount)
    }
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let fc1Parameters = fc1Generator(fc2.flattened())
    let conv2Parameters = conv2Generator(fc1Parameters)
    let conv1Parameters = conv1Generator(conv2Parameters)
    let conv1 = Conv2D<Float>(unflattening: conv1Parameters, like: conv1Base)
    let conv2 = Conv2D<Float>(unflattening: conv2Parameters, like: conv2Base)
    let fc1 = Dense<Float>(unflattening: fc1Parameters, like: fc1Base)
    let convolved = input.sequenced(through: conv1, pool1, conv2, pool2)
    return convolved.sequenced(through: flatten, fc1, fc2)
  }
}

public struct ReverseContextualizedLeNet2: Layer {
  @noDerivative public let conv1Base: Conv2D<Float>
  public var conv1Generator: Sequential<Dense<Float>, Dense<Float>>
  public var pool1: MaxPool2D<Float>
  @noDerivative public let dropout1: Dropout<Float>
  @noDerivative public let conv2Base: Conv2D<Float>
  public var conv2Generator: Sequential<Dense<Float>, Dense<Float>>
  public var pool2: MaxPool2D<Float>
  @noDerivative public let dropout2: Dropout<Float>
  @noDerivative public let conv3Base: Conv2D<Float>
  public var conv3Generator: Sequential<Dense<Float>, Dense<Float>>
  public var pool3: MaxPool2D<Float>
  @noDerivative public let dropout3: Dropout<Float>
  @noDerivative public let conv4Base: Conv2D<Float>
  public var conv4Generator: Sequential<Dense<Float>, Dense<Float>>
  public var pool4: MaxPool2D<Float>
  @noDerivative public let dropout4: Dropout<Float>
  public var flatten: Flatten<Float>
  @noDerivative public let fc1Base: Dense<Float>
  public var fc1Generator: Sequential<Dense<Float>, Dense<Float>>
  @noDerivative public let dropoutFc1: Dropout<Float>
  public var fc2: Dense<Float>

  public init(functionEmbeddingSize: Int) {
    let fc2 = Dense<Float>(inputSize: 1024, outputSize: 10)
    let fc1Base = Dense<Float>(inputSize: 128, outputSize: 1024, activation: gelu)
    self.dropout1 = Dropout<Float>(probability: 0.2)
    self.dropout2 = Dropout<Float>(probability: 0.2)
    self.dropout3 = Dropout<Float>(probability: 0.2)
    self.dropout4 = Dropout<Float>(probability: 0.2)
    self.dropoutFc1 = Dropout<Float>(probability: 0.2)
    self.fc2 = fc2
    self.fc1Base = fc1Base
    self.fc1Generator = Sequential {
      Dense<Float>(inputSize: fc2.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: fc1Base.parameterCount)
    }
    self.flatten = Flatten<Float>()
    self.pool4 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    let conv4Base = Conv2D<Float>(filterShape: (3, 3, 128, 128), padding: .same, activation: gelu)
    self.conv4Base = conv4Base
    self.conv4Generator = Sequential {
      Dense<Float>(inputSize: fc1Base.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: conv4Base.parameterCount)
    }
    self.pool3 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    let conv3Base = Conv2D<Float>(filterShape: (3, 3, 64, 128), padding: .same, activation: gelu)
    self.conv3Base = conv3Base
    self.conv3Generator = Sequential {
      Dense<Float>(inputSize: conv4Base.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: conv3Base.parameterCount)
    }
    self.pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    let conv2Base = Conv2D<Float>(filterShape: (5, 5, 32, 64), padding: .same, activation: gelu)
    self.conv2Base = conv2Base
    self.conv2Generator = Sequential {
      Dense<Float>(inputSize: conv3Base.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: conv2Base.parameterCount)
    }
    self.pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    let conv1Base = Conv2D<Float>(filterShape: (5, 5, 3, 32), padding: .same, activation: gelu)
    self.conv1Base = conv1Base
    self.conv1Generator = Sequential {
      Dense<Float>(inputSize: conv2Base.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: conv1Base.parameterCount)
    }
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let fc1Parameters = fc1Generator(fc2.flattened())
    let conv4Parameters = conv4Generator(fc1Parameters)
    let conv3Parameters = conv3Generator(conv4Parameters)
    let conv2Parameters = conv2Generator(conv3Parameters)
    let conv1Parameters = conv1Generator(conv2Parameters)
    let conv1 = Conv2D<Float>(unflattening: conv1Parameters, like: conv1Base)
    let conv2 = Conv2D<Float>(unflattening: conv2Parameters, like: conv2Base)
    let conv3 = Conv2D<Float>(unflattening: conv3Parameters, like: conv3Base)
    let conv4 = Conv2D<Float>(unflattening: conv4Parameters, like: conv4Base)
    let fc1 = Dense<Float>(unflattening: fc1Parameters, like: fc1Base)
    let convolved1 = input.sequenced(through: conv1, pool1, dropout1, conv2, pool2, dropout2)
    let convolved2 = convolved1.sequenced(through: conv3, pool3, dropout3, conv4, pool4, dropout4)
    return convolved2.sequenced(through: flatten, fc1, dropoutFc1, fc2)
  }
}
