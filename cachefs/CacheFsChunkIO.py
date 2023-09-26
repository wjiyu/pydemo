import threading

from PIL import ContainerIO
import io

from cachefs.CacheFsChunkFile import CacheFsChunkFile


class CacheFsChunkIO(ContainerIO.ContainerIO):

    def __init__(self, tarfile, files=None):
        """
               Create file object.

               :param tarfile: Name of TAR file.
               :param file: Name of member file.
               """
        self.closed = False
        self.fh = open(tarfile, "rb")

        self.members = {}
        self.lock = threading.Lock()

        while True:
            s = self.fh.read(512)

            # read end
            if len(s) < 512:
                break

            if len(s) != 512:
                raise OSError("unexpected end of tar file")

            name = s[:100].decode("utf-8")
            i = name.find("\0")
            if i == 0:
                if not files:
                    break
                else:
                    raise OSError("cannot find subfile")
            if i > 0:
                name = name[:i]

            size = int(s[124:135], 8)

            ###############查找指定内容####################
            # if name in files:
            #     # Open region
            #     self.members[name] = ContainerIO.ContainerIO(self.fh, self.fh.tell(), size)
            #
            # if len(self.members.keys()) == len(files):
            #     break
            ##############################################

            ##################初始化chunk所有文件句柄############################
            if files:
                if name in files:
                    # Open region
                    # super().__init__(self.fh, self.fh.tell(), size)
                    self.members[name] = CacheFsChunkFile(self.fh, self.fh.tell(), size)

                if len(self.members.keys()) == len(files):
                    break
            else:
                self.members[name] = CacheFsChunkFile(self.fh, self.fh.tell(), size)

            self.fh.seek((size + 511) & (~511), io.SEEK_CUR)

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the _Stream object. No operation should be
                   done on it afterwards.
                """
        if self.closed:
            return

        try:
            self.closed = True
        finally:
            self.fh.close()

    def getmembers(self, files, fh=None):
        try:
            members = dict()
            if self.closed or fh:
                for file in files:
                    fh_info = self.members.get(file)
                    fh_info.fh = fh
                    members[file] = fh_info
            else:
                return {file: self.members.get(file) for file in files}

            return members
        except Exception as e:
            print("get members error: ", str(e))
            raise e

    def deletemembers(self, members):
        try:
            for member in members:
                self.members.pop(member, None)
        except Exception as e:
            print("delete members error: ", str(e))
            raise e
