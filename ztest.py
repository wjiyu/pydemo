# import zlib
# import io
# class Zlib:
#     def __init__(self, level):
#         self.level = level
#      def name(self):
#         return "Zlib"
#      def compress(self, src):
#         compress_obj = zlib.compressobj(self.level, zlib.DEFLATED, -zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, 0)
#         compressed_data = compress_obj.compress(src)
#         compressed_data += compress_obj.flush()
#         return compressed_data
#      def decompress(self, src):
#         decompressed_data = zlib.decompress(src, -zlib.MAX_WBITS)
#         return decompressed_data
#  # 示例用法
# zlib_obj = Zlib(level=6)
# data = b"Hello, World!"
# compressed_data = zlib_obj.compress(data)
# print("Compressed Data:", compressed_data)
#  decompressed_data = zlib_obj.decompress(compressed_data)
# print("Decompressed Data:", decompressed_data)