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

public struct ReverseContextualizedLeNet: Layer {
  @noDerivative public var conv1Base: Conv2D<Float>
  public var conv1Generator: Sequential<Dense<Float>, Dense<Float>>
  public var pool1: MaxPool2D<Float>
  @noDerivative public var conv2Base: Conv2D<Float>
  public var conv2Generator: Sequential<Dense<Float>, Dense<Float>>
  public var pool2: MaxPool2D<Float>
  public var flatten: Flatten<Float>
  @noDerivative public var fc1Base: Dense<Float>
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
  @noDerivative public var conv1Base: Conv2D<Float>
  public var conv1Generator: Sequential<Dense<Float>, Dense<Float>>
  public var pool1: MaxPool2D<Float>
  @noDerivative public var conv2Base: Conv2D<Float>
  public var conv2Generator: Sequential<Dense<Float>, Dense<Float>>
  public var pool2: MaxPool2D<Float>
  @noDerivative public var conv3Base: Conv2D<Float>
  public var conv3Generator: Sequential<Dense<Float>, Dense<Float>>
  public var pool3: MaxPool2D<Float>
  public var flatten: Flatten<Float>
  @noDerivative public var fc1Base: Dense<Float>
  public var fc1Generator: Sequential<Dense<Float>, Dense<Float>>
  public var fc2: Dense<Float>

  public init(functionEmbeddingSize: Int) {
    let fc2 = Dense<Float>(inputSize: 1024, outputSize: 10)
    let fc1Base = Dense<Float>(inputSize: 5 * 5 * 128, outputSize: 1024, activation: gelu)
    self.fc2 = fc2
    self.fc1Base = fc1Base
    self.fc1Generator = Sequential {
      Dense<Float>(inputSize: fc2.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
      Dense<Float>(inputSize: functionEmbeddingSize, outputSize: fc1Base.parameterCount)
    }
    self.flatten = Flatten<Float>()
    self.pool3 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    let conv3Base = Conv2D<Float>(filterShape: (5, 5, 64, 128), padding: .same, activation: gelu)
    self.conv3Base = conv3Base
    self.conv3Generator = Sequential {
      Dense<Float>(inputSize: fc1Base.parameterCount, outputSize: functionEmbeddingSize, activation: gelu)
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
    let conv3Parameters = conv3Generator(fc1Parameters)
    let conv2Parameters = conv2Generator(conv3Parameters)
    let conv1Parameters = conv1Generator(conv2Parameters)
    let conv1 = Conv2D<Float>(unflattening: conv1Parameters, like: conv1Base)
    let conv2 = Conv2D<Float>(unflattening: conv2Parameters, like: conv2Base)
    let conv3 = Conv2D<Float>(unflattening: conv3Parameters, like: conv3Base)
    let fc1 = Dense<Float>(unflattening: fc1Parameters, like: fc1Base)
    let convolved = input.sequenced(through: conv1, pool1, conv2, pool2, conv3, pool3)
    return convolved.sequenced(through: flatten, fc1, fc2)
  }
}
