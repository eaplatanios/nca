// // Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License"); you may not
// // use this file except in compliance with the License. You may obtain a copy of
// // the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// // WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// // License for the specific language governing permissions and limitations under
// // the License.

// public struct Atom: Hashable {
//   public let id: String
//   public let type: Type

//   public init(id: String, type: Type) {
//     self.id = id
//     self.type = type
//   }
// }

// extension Atom {
//   public enum Type: Int, Hashable {
//     case Person = 0
//     case Building = 1
//     case Cell = 2
//   }
// }

// public enum Predicate: Int, Hashable {
//   case house = 0
//   case commonBuilding = 1
//   case livesIn = 2
//   case locatedAt = 3

//   public var kind: Kind {
//     switch self {
//     case .house, .commonBuilding, .livesIn, .locatedAt: return .location
//     }
//   }
// }

// extension Predicate {
//   public enum Kind: Int, Hashable {
//     case location = 0
//     case relationship = 1
//   }
// }

// public struct PredicateCollection {
//   public let predicates: Set<Predicate>
// }

// public protocol Rule {
//   func apply(on predicates: Set<Predicate>) -> Set<Predicate>
// }
