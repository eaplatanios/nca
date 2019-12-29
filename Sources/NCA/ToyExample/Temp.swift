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

// // People: Alice, Bob, Eve, ...
// // Locations: House[<person>...], Library, Playground, ...
// // Predicates:
// // - House[<person>] -> <location>
// // - WestOf[<location>] -> <location>
// // - WestOf[<location>, <location>] -> <boolean>
// // - NeighborOf[<person>] -> <person>
// // - NeighborOf[<person>, <person>] -> <boolean>
// // - Friends[<person>, <person>] -> <boolean>
// // - Mom[<person>] -> <person>
// // - Dad[<person>] -> <person>
// // - Siblings[<person>] -> [<person>]
// // - Friends[<person>] -> [<person>]
// // - Housemates[<person>] -> [<person>]
// // - Housemates[[<person>]] -> <boolean>

// // Person:
// // - Mom
// // - Dad
// // - Siblings
// // - Friends
// // - House Location (multiple people can live in the same house.

// // Common Sense Axioms (not explicitly provided---need to be learned):
// // - relativeCellToCell(???)     [Coordinate System]
// // - mother(x, y) -> female(y)
// // - father(x, y) -> male(y)
// // - sister(x, y) -> female(y)
// // - brother(x, y) -> male(y)

// // People's names may differ from one example to the next.

// // Problems:
// // - Is a given logical expression true? The expression could involve quantifiers.
// // - Missing variable(s) in logical expression.

// // Input Modalities:
// // - Partial/Full view of the map
// // - Logical Expressions

// // Output Modalities:
// // - Yes/No/Maybe answer
// // - Person classifier/labeler
// // - Location classifier/labeler
// // - Cell generator

// public enum Rule {
//   case livesAtExpansion
//   case relativeLocationExpansion

//   public var complexityLevel: Int {
//     switch self {
//     case .livesAtExpansion: return 0
//     case .relativeLocationExpansion: return 0
//     }
//   }

//   public func implications(
//     predicate: Predicate,
//     using map: Map,
//     theory: Set<Predicate>,
//     complexityLevel: Int
//   ) -> Set<Predicate> {
//     if self.complexityLevel > complexityLevel { return [] }
//     switch (self, predicate) {
//     case (.livesAtExpansion, let .livesAtCell(person, cell)):
//       guard let location = map.locations[cell] else { return [] }
//       return [predicate, .livesAtLocation(person, location)]
//     case (.livesAtExpansion, let .livesAtLocation(person, location)):
//       guard let cell = map.cells[location] else { return [] }
//       return [predicate, .livesAtCell(person, cell)]
//     case (.relativeLocationExpansion, let .relativeLocation(l1, d1, l2)):
//       let c1 = map.cells[l1]!
//       let c2 = map.cells[l2]!
//       let d2 = d1.opposite
//       let c2s = map.cells.values.filter { $0.relative(to: c1, direction: d1) }
//       let c1s = map.cells.values.filter { $0.relative(to: c2, direction: d2) }
//       let lp1s = c2s.map { map.locations[$0]! }.map { Predicate.relativeLocation(l1, d1, $0) }
//       let lp2s = c1s.map { map.locations[$0]! }.map { Predicate.relativeLocation(l2, d2, $0) }
//       let cp1s = c2s.map { Predicate.relativeLocationToCell(l1, d1, $0) }
//       let cp2s = c1s.map { Predicate.relativeLocationToCell(l2, d2, $0) }
//       return Set([predicate] + lp1s + lp2s + cp1s + cp2s)
//     case (.relativeLocationExpansion, let .relativeLocationToCell(l1, d1, c2)):
//       let c1 = map.cells[l1]!
//       let l2 = map.locations[c1]!
//       let d2 = d1.opposite
//       let c2s = map.cells.values.filter { $0.relative(to: c1, direction: d1) }
//       let c1s = map.cells.values.filter { $0.relative(to: c2, direction: d2) }
//       let lp1s = c2s.map { Predicate.relativeLocationToCell(l1, d1, $0) }
//       let lp2s = c1s.map { Predicate.relativeLocationToCell(l2, d2, $0) }
//       let cp1s = c2s.map { Predicate.relativeLocationToCell(l1, d1, $0) }
//       let cp2s = c1s.map { Predicate.relativeLocationToCell(l2, d2, $0) }
//       return Set([predicate] + lp1s + lp2s + cp1s + cp2s)
//     default: return [predicate]
//     }
//   }
// }

// // TODO: What about existentials? E.g., ∃l.lives(.person("John"), l) AND .located(l, west, "Library")
// public enum Predicate: Hashable {
//   case livesAtCell(Person, Cell)                         // livesAtCell(John, Cell(2, 3))
//   case livesAtLocation(Person, Location)                 // livesAtLocation(John, House[Jake])
//   case relativeLocation(Location, Direction, Location)   // relativeLocation(Building[Library], west (of), Person[John])
//   case relativeLocationToCell(Location, Direction, Cell) // relativeLocationToCell(Building[Library], west (of), Cell(-1, 0))
//   case neighboringLocations(Location, Location)          // neighboringLocations(Building[Library], House[Jake])
//   case neighboringPeople(Person, Person)                 // neighboringPeople(Building[Library], House[Jake])
//   case friend(Person, Person)
//   case parent(Person, Person)
//   case mother(Person, Person)
//   case father(Person, Person)
//   case sibling(Person, Person)
//   case sister(Person, Person)
//   case brother(Person, Person)
//   case male(Person)
//   case female(Person)

// //  public func evaluate(for map: Map) -> Bool {
// //    switch self {
// //    case let .lives(person, location):
// //    }
// //  }
// }

// public struct Person: Hashable {
//   public let name: String

//   public init(name: String) {
//     self.name = name
//   }
// }

// public enum Location: Hashable {
//   case house([Person])
//   case building(String)

//   public var name: String {
//     switch self {
//     case let .house(people): return "House[\(people.map { $0.name }.joined(separator: ", "))]"
//     case let .building(name): return "Building[\(name)]"
//     }
//   }
// }

// public enum Direction {
//   case north, west, south, east

//   public var opposite: Direction {
//     switch self {
//     case .north: return .south
//     case .west: return .east
//     case .south: return .north
//     case .east: return .west
//     }
//   }
// }

// public struct Cell: Hashable {
//   public let x: Int
//   public let y: Int

//   public var west: Cell { Cell(x - 1, y) }
//   public var east: Cell { Cell(x + 1, y) }
//   public var north: Cell { Cell(x, y + 1) }
//   public var south: Cell { Cell(x, y - 1) }

//   public var neighbors: [Cell] {
//     [Cell(x + 1, y), Cell(x - 1, y), Cell(x, y + 1), Cell(x, y - 1)]
//   }

//   public init(_ x: Int, _ y: Int) {
//     self.x = x
//     self.y = y
//   }

//   public func westOf(_ other: Cell) -> Bool { x < other.x }
//   public func eastOf(_ other: Cell) -> Bool { x > other.x }
//   public func northOf(_ other: Cell) -> Bool { y > other.y }
//   public func southOf(_ other: Cell) -> Bool { y < other.y }

//   public func relative(to other: Cell, direction: Direction) -> Bool {
//     switch direction {
//     case .west: return westOf(other)
//     case .east: return eastOf(other)
//     case .north: return northOf(other)
//     case .south: return southOf(other)
//     }
//   }
// }

// public struct Map {
//   public let cells: [Location: Cell]
//   public let locations: [Cell: Location]

//   public init(cells: [Location: Cell]) {
//     self.cells = cells
//     self.locations = [Cell: Location](uniqueKeysWithValues: cells.map { ($0.value, $0.key) })
//   }

//   public init(locations: [Cell: Location]) {
//     self.cells = [Location: Cell](uniqueKeysWithValues: locations.map { ($0.value, $0.key) })
//     self.locations = locations
//   }
// }

// //===-----------------------------------------------------------------------------------------===//
// // Map Generation
// //===-----------------------------------------------------------------------------------------===//

// extension Map {
//   public static func generate<T: RandomNumberGenerator>(
//     for locations: [Location],
//     using generator: inout T
//   ) -> Map {
//     let locations = locations.shuffled()
//     var availableCells = Set<Cell>([Cell(0, 0)])
//     var occupiedCells = Set<Cell>()
//     let cells = locations.map { location -> (Location, Cell) in
//       let cell = availableCells.randomElement(using: &generator)!
//       availableCells.remove(cell)
//       occupiedCells.update(with: cell)

//       // Add newly available cells.
//       for neighbor in cell.neighbors {
//         if !occupiedCells.contains(neighbor) {
//           availableCells.update(with: neighbor)
//         }
//       }

//       return (location, cell)
//     }
//     return Map(cells: [Location: Cell](uniqueKeysWithValues: cells))
//   }
// }

// extension Map {
//   public static func generate(for locations: [Location]) -> Map {
//     var generator = SystemRandomNumberGenerator()
//     return generate(for: locations, using: &generator)
//   }
// }

// //===-----------------------------------------------------------------------------------------===//
// // Map Rendering
// //===-----------------------------------------------------------------------------------------===//

// extension Map {
//   public var rendering: String {
//     if locations.isEmpty { return "" }
//     let cellWidth = locations.values.map { $0.name.count + 2 }.max()!
//     let cells = self.cells.values
//     let westBoundary = cells.map { $0.x }.min()!
//     let eastBoundary = cells.map { $0.x }.max()!
//     let northBoundary = -cells.map { $0.y }.max()!
//     let southBoundary = -cells.map { $0.y }.min()!
//     var result = "┌"
//     for _ in westBoundary..<eastBoundary {
//       result += "\(String(repeating: "─", count: cellWidth))┬"
//     }
//     result += "\(String(repeating: "─", count: cellWidth))┐\n│"
//     for x in westBoundary...eastBoundary {
//       let cell = Cell(x, -northBoundary)
//       if let location = locations[cell] {
//         result += "\(location: location, width: cellWidth)│"
//       } else {
//         result += "\(String(repeating: " ", count: cellWidth))│"
//       }
//     }
//     result += "\n"
//     for y in (northBoundary + 1)..<southBoundary {
//       result += "├"
//       for _ in westBoundary..<eastBoundary {
//         result += "\(String(repeating: "─", count: cellWidth))┼"
//       }
//       result += "\(String(repeating: "─", count: cellWidth))┤\n│"
//       for x in westBoundary...eastBoundary {
//         let cell = Cell(x, -y)
//         if let location = locations[cell] {
//           result += "\(location: location, width: cellWidth)│"
//         } else {
//           result += "\(String(repeating: " ", count: cellWidth))│"
//         }
//       }
//       result += "\n"
//     }
//     result += "└"
//     for _ in westBoundary..<eastBoundary {
//       result += "\(String(repeating: "─", count: cellWidth))┴"
//     }
//     result += "\(String(repeating: "─", count: cellWidth))┘"
//     return result
//   }
// }

// extension String.StringInterpolation {
//   fileprivate mutating func appendInterpolation(location: Location, width: Int = 10) {
//     var literal = String(location.name.prefix(width))
//     let padding = width - literal.count
//     if padding > 0 {
//       let rightPadding = padding / 2
//       let leftPadding = padding - rightPadding
//       literal = String(repeating: " ", count: leftPadding) + literal
//       literal += String(repeating: " ", count: rightPadding)
//     }
//     appendLiteral(literal)
//   }
// }
