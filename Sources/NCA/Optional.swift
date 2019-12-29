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

extension Optional: Differentiable where Wrapped: Differentiable {
  public struct TangentVector:  Differentiable, AdditiveArithmetic {
    public typealias TangentVector = Self

    public var value: Wrapped.TangentVector?

    public init(value: Wrapped.TangentVector?) {
      self.value = value
    }

    public static var zero: Self { Self(value: Wrapped.TangentVector.zero) }

    public static func + (lhs: Self, rhs: Self) -> Self {
      switch (lhs.value, rhs.value) {
      case (.none, .none): return Optional.TangentVector(value: .none)
      case (_, .none): return lhs
      case (.none, _): return rhs
      case let (.some(x), .some(y)): return Optional.TangentVector(value: .some(x + y))
      }
    }

    public static func - (lhs: Self, rhs: Self) -> Self {
      switch (lhs.value, rhs.value) {
      case (.none, .none): return Optional.TangentVector(value: .none)
      case (_, .none): return lhs
      case let (.none, .some(y)):
        return Optional.TangentVector(value: .some(Wrapped.TangentVector.zero - y))
      case let (.some(x), .some(y)):
        return Optional.TangentVector(value: .some(x - y))
      }
    }
  }

  public var zeroTangentVector: TangentVector {
    TangentVector(value: .zero)
  }

  public mutating func move(along direction: TangentVector) {
    self?.move(along: direction.value!)
  }
}
