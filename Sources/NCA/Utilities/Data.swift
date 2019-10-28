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
import Logging
import Progress

extension IteratorProtocol {
  /// Returns an iterator that maps elements of this iterator using the provided function.
  ///
  /// - Parameters:
  ///   - mapFn: Function used to map the iterator elements.
  public func map<MappedElement>(
    _ mapFn: @escaping (Element) -> MappedElement
  ) -> MapIterator<Self, MappedElement> {
    MapIterator(self, mapFn: mapFn)
  }

  /// Returns an iterator that repeats this iterator indefinitely.
  public func repeated() -> RepeatIterator<Self> {
    RepeatIterator(self)
  }

  /// Returns an iterator that shuffles this iterator using a temporary buffer.
  ///
  /// - Parameters:
  ///   - bufferSize: Size of the shuffle buffer.
  public func shuffled(bufferSize: Int) -> ShuffleIterator<Self> {
    ShuffleIterator(self, bufferSize: bufferSize)
  }

  // TODO: [DOC] Add documentation string.
  public func grouped(
    keyFn: @escaping (Element) -> Int,
    sizeFn: @escaping (Int) -> Int,
    reduceFn: @escaping ([Element]) -> Element
  ) -> GroupedIterator<Self> {
    GroupedIterator(self, keyFn: keyFn, sizeFn: sizeFn, reduceFn: reduceFn)
  }
}

extension IteratorProtocol where Element: KeyPathIterable {
  /// Returns an iterator that batches elements of this iterator.
  ///
  /// - Parameters:
  ///   - batchSize: Batch size.
  public func batched(batchSize: Int) -> BatchIterator<Self> {
    BatchIterator(self, batchSize: batchSize)
  }
}

/// Iterator that maps elements of another iterator using the provided function.
public struct MapIterator<Base: IteratorProtocol, MappedElement>: IteratorProtocol {
  private var iterator: Base
  private let mapFn: (Base.Element) -> MappedElement

  public init(_ iterator: Base, mapFn: @escaping (Base.Element) -> MappedElement) {
    self.iterator = iterator
    self.mapFn = mapFn
  }

  public mutating func next() -> MappedElement? {
    if let element = iterator.next() { return mapFn(element) }
    return nil
  }
}

/// Iterator that repeats another iterator indefinitely.
public struct RepeatIterator<Base: IteratorProtocol>: IteratorProtocol {
  private let originalIterator: Base
  private var currentIterator: Base

  public init(_ iterator: Base) {
    self.originalIterator = iterator
    self.currentIterator = iterator
  }

  public mutating func next() -> Base.Element? {
    if let element = currentIterator.next() {
      return element
    }
    currentIterator = originalIterator
    return currentIterator.next()
  }
}

/// Iterator that shuffles another iterator using a temporary buffer.
public struct ShuffleIterator<Base: IteratorProtocol>: IteratorProtocol {
  private let bufferSize: Int
  private var iterator: Base
  private var buffer: [Base.Element]
  private var bufferIndex: Int

  public init(_ iterator: Base, bufferSize: Int) {
    self.bufferSize = bufferSize
    self.iterator = iterator
    self.buffer = []
    self.bufferIndex = 0
  }

  public mutating func next() -> Base.Element? {
    if buffer.isEmpty || (bufferIndex >= bufferSize && bufferSize != -1) { fillBuffer() }
    if buffer.isEmpty { return nil }
    bufferIndex += 1
    return buffer[bufferIndex - 1]
  }

  private mutating func fillBuffer() {
    buffer = []
    bufferIndex = 0
    while let element = iterator.next(), bufferIndex < bufferSize || bufferSize == -1 {
      buffer.append(element)
      bufferIndex += 1
    }
    bufferIndex = 0
  }
}

/// Iterator that batches elements from another iterator.
public struct BatchIterator<Base: IteratorProtocol>: IteratorProtocol
where Base.Element: KeyPathIterable {
  private let batchSize: Int
  private var iterator: Base
  private var buffer: [Base.Element]

  public init(_ iterator: Base, batchSize: Int) {
    self.batchSize = batchSize
    self.iterator = iterator
    self.buffer = []
    self.buffer.reserveCapacity(batchSize)
  }

  public mutating func next() -> Base.Element? {
    while buffer.count < batchSize {
      if let element = iterator.next() {
        buffer.append(element)
      } else {
        break
      }
    }
    if buffer.isEmpty { return nil }
    let batch = Base.Element.batch(buffer)
    buffer = []
    buffer.reserveCapacity(batchSize)
    return batch
  }
}

/// Iterator that grouped elements from another iterator.
public struct GroupedIterator<Base: IteratorProtocol>: IteratorProtocol {
  private let keyFn: (Base.Element) -> Int
  private let sizeFn: (Int) -> Int
  private let reduceFn: ([Base.Element]) -> Base.Element
  private var iterator: Base
  private var groups: [Int: [Base.Element]]

  public init(
    _ iterator: Base,
    keyFn: @escaping (Base.Element) -> Int,
    sizeFn: @escaping (Int) -> Int,
    reduceFn: @escaping ([Base.Element]) -> Base.Element
  ) {
    self.keyFn = keyFn
    self.sizeFn = sizeFn
    self.reduceFn = reduceFn
    self.iterator = iterator
    self.groups = [Int: [Base.Element]]()
  }

  public mutating func next() -> Base.Element? {
    var elements: [Base.Element]? = nil
    while elements == nil {
      if let element = iterator.next() {
        let key = keyFn(element)
        if !groups.keys.contains(key) {
          groups[key] = [element]
        } else {
          groups[key]!.append(element)
        }
        if groups[key]!.count >= sizeFn(key) {
          elements = groups.removeValue(forKey: key)!
        }
      } else {
        break
      }
    }
    guard let elementsToReduce = elements else { return nil }
    return reduceFn(elementsToReduce)
  }
}

#if os(Linux)
import FoundationNetworking
#endif

/// Downloads the file at `url` to `path`, if `path` does not exist.
///
/// - Parameters:
///   - from: URL to download data from.
///   - to: Destination file path.
///
/// - Returns: Boolean value indicating whether a download was
///     performed (as opposed to not needed).
internal func maybeDownload(from url: URL, to destination: URL) throws {
  if !FileManager.default.fileExists(atPath: destination.path) {
    // Create any potentially missing directories.
    try FileManager.default.createDirectory(
      atPath: destination.deletingLastPathComponent().path,
      withIntermediateDirectories: true)

    // Create the URL session that will be used to download the dataset.
    let semaphore = DispatchSemaphore(value: 0)
    let delegate = DataDownloadDelegate(destinationFileUrl: destination, semaphore: semaphore)
    let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

    // Download the data to a temporary file and then copy that file to
    // the destination path.
    logger.info("Downloading \(url).")
    let task = session.downloadTask(with: url)
    task.resume()

    // Wait for the download to finish.
    semaphore.wait()
  }
}

internal class DataDownloadDelegate: NSObject, URLSessionDownloadDelegate {
  let destinationFileUrl: URL
  let semaphore: DispatchSemaphore
  let numBytesFrequency: Int64

  internal var logCount: Int64 = 0
  internal var progressBar: ProgressBar? = nil

  init(
    destinationFileUrl: URL,
    semaphore: DispatchSemaphore,
    numBytesFrequency: Int64 = 1024 * 1024
  ) {
    self.destinationFileUrl = destinationFileUrl
    self.semaphore = semaphore
    self.numBytesFrequency = numBytesFrequency
  }

  internal func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64,
    totalBytesExpectedToWrite: Int64
  ) -> Void {
    if progressBar == nil {
      progressBar = ProgressBar(
        count: Int(totalBytesExpectedToWrite) / (1024 * 1024),
        configuration: [
          ProgressString(string: "Download Progress (MBs):"),
          ProgressIndex(),
          ProgressBarLine(),
          ProgressTimeEstimates()])
    }
    progressBar!.setValue(Int(totalBytesWritten) / (1024 * 1024))
  }

  internal func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) -> Void {
    do {
      try FileManager.default.moveItem(at: location, to: destinationFileUrl)
    } catch (let writeError) {
      logger.error("Error writing file \(location.path) : \(writeError)")
    }
    logger.info("Downloaded successfully to \(location.path).")
    semaphore.signal()
  }
}

internal func extract(zipFileAt source: URL, to destination: URL) throws {
  logger.info("Extracting file at '\(source.path)'.")
  let process = Process()
  process.environment = ProcessInfo.processInfo.environment
  process.executableURL = URL(fileURLWithPath: "/bin/bash")
  process.arguments = ["-c", "unzip -d \(destination.path) \(source.path)"]
  try process.run()
  process.waitUntilExit()
}

internal func extract(tarGZippedFileAt source: URL, to destination: URL) throws {
  logger.info("Extracting file at '\(source.path)'.")
  try FileManager.default.createDirectory(
    at: destination,
    withIntermediateDirectories: false)
  let process = Process()
  process.environment = ProcessInfo.processInfo.environment
  process.executableURL = URL(fileURLWithPath: "/bin/bash")
  process.arguments = ["-c", "tar -C \(destination.path) -xzf \(source.path)"]
  try process.run()
  process.waitUntilExit()
}

internal func parse(tsvFileAt fileURL: URL) throws -> [[String]] {
  try Data(contentsOf: fileURL).withUnsafeBytes {
    $0.split(separator: UInt8(ascii: "\n")).map {
      $0.split(separator: UInt8(ascii: "\t"), omittingEmptySubsequences: false)
        .map { String(decoding: UnsafeRawBufferPointer(rebasing: $0), as: UTF8.self) }
    }
  }
}
