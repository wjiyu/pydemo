from PIL import ContainerIO
import io


class CacheFsTarIO(ContainerIO.ContainerIO):

    def __init__(self, tarfile, files):
        """
               Create file object.

               :param tarfile: Name of TAR file.
               :param file: Name of member file.
               """
        self.closed = False
        self.fh = open(tarfile, "rb")

        self.members = dict()

        while True:
            s = self.fh.read(512)

            # read end
            # if len(s) < 512:
            #     break

            if len(s) != 512:
                raise OSError("unexpected end of tar file")

            name = s[:100].decode("utf-8")
            i = name.find("\0")
            if i == 0:
                # if not files:
                #     break
                # else:
                raise OSError("cannot find subfile")
            if i > 0:
                name = name[:i]

            size = int(s[124:135], 8)

            if name in files:
                # Open region
                self.members[name] = ContainerIO.ContainerIO(self.fh, self.fh.tell(), size)

            if len(self.members.keys()) == len(files):
                break

            # if files:
            #     if name in files:
            #         # Open region
            #         # super().__init__(self.fh, self.fh.tell(), size)
            #         self.members[name] = ContainerIO.ContainerIO(self.fh, self.fh.tell(), size)
            #
            #     if len(self.members.keys()) == len(files):
            #         break
            # else:
            #     self.members[name] = ContainerIO.ContainerIO(self.fh, self.fh.tell(), size)

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

    def getmembers(self):
        return self.members
