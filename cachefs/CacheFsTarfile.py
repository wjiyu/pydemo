import tarfile

class CacheFsTarfile(tarfile.TarFile):

    OPEN_METH = {
        "tar": "taropen"   # uncompressed tar
    }
    def __init__(self, name=None, mode="r", fileobj=None, format=None,
                 tarinfo=None, dereference=None, ignore_zeros=None, encoding=None,
                 errors="surrogateescape", pax_headers=None, debug=None,
                 errorlevel=None, copybufsize=None):

        self.member_maps = {}

        super().__init__(name, mode, fileobj, format, tarinfo, dereference, ignore_zeros, encoding,
                                             errors, pax_headers, debug, errorlevel, copybufsize)

        # if mode in "r":
        #     self.getmembers()
        #

    @classmethod
    def open(cls, name=None, mode="r", fileobj=None, bufsize=tarfile.RECORDSIZE, **kwargs):

        if not name and not fileobj:
            raise ValueError("nothing to open")

        if mode in ("r", "r:*"):
            # Find out which *open() is appropriate for opening the file.
            def not_compressed(comptype):
                return cls.OPEN_METH[comptype] == 'taropen'

            for comptype in sorted(cls.OPEN_METH, key=not_compressed):
                func = getattr(cls, cls.OPEN_METH[comptype])
                if fileobj is not None:
                    saved_pos = fileobj.tell()
                try:
                    return func(name, "r", fileobj, **kwargs)
                except (tarfile.ReadError, tarfile.CompressionError):
                    if fileobj is not None:
                        fileobj.seek(saved_pos)
                    continue
            raise tarfile.ReadError("file could not be opened successfully")

        elif ":" in mode:
            filemode, comptype = mode.split(":", 1)
            filemode = filemode or "r"
            comptype = comptype or "tar"

            # Select the *open() function according to
            # given compression.
            if comptype in cls.OPEN_METH:
                func = getattr(cls, cls.OPEN_METH[comptype])
            else:
                raise tarfile.CompressionError("unknown compression type %r" % comptype)
            return func(name, filemode, fileobj, **kwargs)

        elif "|" in mode:
            filemode, comptype = mode.split("|", 1)
            filemode = filemode or "r"
            comptype = comptype or "tar"

            if filemode not in ("r", "w"):
                raise ValueError("mode must be 'r' or 'w'")

            stream = tarfile._Stream(name, filemode, comptype, fileobj, bufsize)
            try:
                t = cls(name, filemode, stream, **kwargs)
            except:
                stream.close()
                raise
            t._extfileobj = False
            return t

        elif mode in ("a", "w", "x"):
            return cls.taropen(name, mode, fileobj, **kwargs)

        raise ValueError("undiscernible mode")

    def extractfile(self, member):
        self._check("r")

        if isinstance(member, str):
            tarinfo = self.member_maps.get(member)
        else:
            tarinfo = member

        if tarinfo.isreg() or tarinfo.type not in tarfile.SUPPORTED_TYPES:
            # Members with unknown types are treated as regular files.
            return self.fileobject(self, tarinfo)

        elif tarinfo.islnk() or tarinfo.issym():
            if isinstance(self.fileobj, tarfile._Stream):
                # A small but ugly workaround for the case that someone tries
                # to extract a (sym)link as a file-object from a non-seekable
                # stream of tar blocks.
                raise tarfile.StreamError("cannot extract (sym)link as file object")
            else:
                # A (sym)link's file object is its target's file object.
                return self.extractfile(self._find_link_target(tarinfo))
        else:
            # If there's no data associated with the member (directory, chrdev,
            # blkdev, etc.), return None instead of a file object.
            return None

    def extractfiles(self, members):
        self._check("r")

        member_maps = self.getmembermaps()

        if isinstance(members, list):
            tarinfos = {member: member_maps.get(member) for member in members}
        elif isinstance(members, str):
            tarinfos = self.member_maps.get(members)
        else:
            tarinfos = members

        fileobjects = dict()
        for member, tarinfo in tarinfos.items():
            if tarinfo.isreg() or tarinfo.type not in tarfile.SUPPORTED_TYPES:
                # Members with unknown types are treated as regular files.
                fileobjects[member] = self.fileobject(self, tarinfo)

            elif tarinfo.islnk() or tarinfo.issym():
                if isinstance(self.fileobj, tarfile._Stream):
                    # A small but ugly workaround for the case that someone tries
                    # to extract a (sym)link as a file-object from a non-seekable
                    # stream of tar blocks.
                    raise tarfile.StreamError("cannot extract (sym)link as file object")
                else:
                    # A (sym)link's file object is its target's file object.
                    return self.extractfile(self._find_link_target(tarinfo))
            else:
                # If there's no data associated with the member (directory, chrdev,
                # blkdev, etc.), return None instead of a file object.
                return None
        return fileobjects


    def next(self):
        """Return the next member of the archive as a TarInfo object, when
           TarFile is opened for reading. Return None if there is no more
           available.
        """
        self._check("ra")
        if self.firstmember is not None:
            m = self.firstmember
            self.firstmember = None
            return m

        # Advance the file pointer.
        if self.offset != self.fileobj.tell():
            self.fileobj.seek(self.offset - 1)
            if not self.fileobj.read(1):
                raise tarfile.ReadError("unexpected end of data")

        # Read the next block.
        tarinfo = None
        while True:
            try:
                tarinfo = self.tarinfo.fromtarfile(self)
            except tarfile.EOFHeaderError as e:
                if self.ignore_zeros:
                    self._dbg(2, "0x%X: %s" % (self.offset, e))
                    self.offset += tarfile.BLOCKSIZE
                    continue
            except tarfile.InvalidHeaderError as e:
                if self.ignore_zeros:
                    self._dbg(2, "0x%X: %s" % (self.offset, e))
                    self.offset += tarfile.BLOCKSIZE
                    continue
                elif self.offset == 0:
                    raise tarfile.ReadError(str(e))
            except tarfile.EmptyHeaderError:
                if self.offset == 0:
                    raise tarfile.ReadError("empty file")
            except tarfile.TruncatedHeaderError as e:
                if self.offset == 0:
                    raise tarfile.ReadError(str(e))
            except tarfile.SubsequentHeaderError as e:
                raise tarfile.ReadError(str(e))
            break

        if tarinfo is not None:
            self.members.append(tarinfo)
            self.member_maps[tarinfo.name] = tarinfo
        else:
            self._loaded = True

        return tarinfo

    def getmembers(self):
        """Return the members of the archive as a list of TarInfo objects. The
           list has the same order as the members in the archive.
        """
        self._check()
        if not self._loaded:  # if we want to obtain a list of
            self._load()  # all members, we first have to
            # scan the whole archive.
        return self.members

    def getmembermaps(self):
        """Return the members of the archive as a list of TarInfo objects. The
           list has the same order as the members in the archive.
        """
        self._check()
        if not self._loaded:  # if we want to obtain a list of
            self._load()  # all members, we first have to
            # scan the whole archive.
        return self.member_maps

    def _load(self):
        """Read through the entire archive file and look for readable
           members.
        """
        while True:
            tarinfo = self.next()
            if tarinfo is None:
                break
        self._loaded = True

    @classmethod
    def taropen(cls, name, mode="r", fileobj=None, **kwargs):
        """Open uncompressed tar archive name for reading or writing.
        """
        if len(mode) > 1 or mode not in "raw":
            raise ValueError("mode must be 'r', 'a' or 'w'")
        return cls(name, mode, fileobj, **kwargs)

