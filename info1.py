# from typing import List, Dict, Tuple
# import math
# import json
# import zlib
# import threading
#
# from tensorflow import uint64
#
#
# class namedNode:
#     def __init__(self, node, Name, Chunkid):
#         self.node = node
#         self.Name = Name
#         self.Chunkid = Chunkid
# class chunkFile:
#     def __init__(self, Id, Inode, ChunkId, Name):
#         self.Id = Id
#         self.Inode = Inode
#         self.ChunkId = ChunkId
#         self.Name = Name
# class ChunkInfo:
#     def __init__(self):
#         self.chunkMap = {}
#         self.Files = []
#         self.mux = threading.Lock()
# class dbMeta:
#     def GetChunkMetaInfo(self, inode, name, isDir, work, compression) -> Tuple[Dict[uint64, List[str]], List[str], Exception]:
#         nodes = []
#         try:
#             if isDir:
#                 # query nodes for directories
#                 if name != "":
#                     # apply name filter
#                     pass
#             else:
#                 # query nodes for files
#                 pass
#         except Exception as err:
#             logger.error(f"query meta info error: {err}")
#             return None, None, err
#          try:
#             infos = self.ScanInfos(nodes, name, work, compression)
#             return infos.chunkMap, infos.Files, None
#         except Exception as err:
#             return None, None, err
#      def SplitArray(self, nodes, number):
#         if nodes is None:
#             return None
#         if number <= 0:
#             return None
#
#         size = math.ceil(len(nodes) / number)
#         slices = [nodes[i:i+size] for i in range(0, len(nodes), size)]
#
#         return slices
#
#      def ScanInfos(self, nodes, name, work, compression) -> Tuple[ChunkInfo, Exception]:
#         slices = self.SplitArray(nodes, work)
#         if slices is None:
#             return None, None
#          info = ChunkInfo()
#         threads = []
#         for i in range(work):
#             if i > len(slices) - 1:
#                 break
#             t = threading.Thread(target=self.ParsingFiles, args=(slices[i], name, info, compression))
#             threads.append(t)
#             t.start()
#          for t in threads:
#             t.join()
#          return info, None
#      def ParsingFiles(self, nodes, name, info, compression):
#         chunkIds = []
#         for node in nodes:
#             if len(node.Name) == 0:
#                 logger.error(f"Corrupt entry with empty name: inode {node.Inode}")
#                 continue
#             logger.debug(f"name: {node.Name}")
#             index = node.Name.rfind("_")
#             if index < 0:
#                 logger.debug(f"filter non dataset info: {node.Name}")
#                 continue
#             subName = node.Name[:index]
#              if name != "":
#                 if name == subName:
#                     chunkIds.append(node.Chunkid)
#             else:
#                 chunkIds.append(node.Chunkid)
#          fileList = []
#         chunkMap = {}
#         zlibCompress = zlib.decompressobj(level=zlib.BEST_COMPRESSION)
#         try:
#             # query slice files
#             for chunkId in chunkIds:
#                 # process each item in the batch
#                 files = []
#                 if compression:
#                     # decompress files
#                     pass
#                 else:
#                     # no compression
#                     pass
#                 fileList.extend(files)
#                 chunkMap[chunkId] = files
#         except Exception as err:
#             logger.error(err)
#          info.mux.acquire()
#         info.Files.extend(fileList)
#         info.chunkMap.update(chunkMap)
#         info.mux.release()