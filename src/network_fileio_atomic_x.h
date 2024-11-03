#ifndef __DG_NETWORK_FILEIO_ATOMIC_X_H__
#define __DG_NETWORK_FILEIO_ATOMIC_X_H__

//atomic is only required for metadatas - chksum and unified
//atomic guarantees that a failed write operation guarantees the next read reads either the old data or the new one
//atomic guarantees that a success write operation guarantees the next read to read the written data (this is not chksum responsibility - due to cache + fsync + friends - yeah - that's the twist) - chksum only guarantees to a success read to return a success write at a random point in time  


#endif