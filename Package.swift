// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "NCA",
  products: [
    .library(name: "NCA", targets: ["NCA"]),
    .executable(name: "Experiments", targets: ["Experiments"])
  ],
  dependencies: [
    .package(url: "https://github.com/apple/swift-log.git", from: "1.0.0"),
    .package(url: "https://github.com/apple/swift-package-manager.git", from: "0.4.0"),
    .package(url: "https://github.com/jkandzi/Progress.swift.git", from: "0.4.0"),
    .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.6.0")
  ],
  targets: [
    .target(name: "NCA", dependencies: ["Logging", "Progress", "SwiftProtobuf"]),
    .target(name: "Experiments", dependencies: ["NCA"]),
    .testTarget(name: "NCATests", dependencies: ["NCA"])
  ]
)
