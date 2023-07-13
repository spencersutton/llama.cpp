// Internal header to be included only by llama.cpp.
// Contains wrappers around OS interfaces.

#ifndef LLAMA_UTIL_H
#define LLAMA_UTIL_H

#include <cstdio>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <cstdarg>
#include <cstdlib>
#include <climits>

#include <string>
#include <vector>
#include <stdexcept>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#define LLAMA_ASSERT(x)                                                           \
    do {                                                                          \
        if (!(x)) {                                                               \
            fprintf(stderr, "LLAMA_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort();                                                              \
        }                                                                         \
    } while (0)

#ifdef __GNUC__
__attribute__((format(printf, 1, 2)))
#endif
static std::string
format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    LLAMA_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    LLAMA_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

struct llama_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

    llama_file(const char * fname, const char * mode) {
        fp = std::fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        long ret = std::ftell(fp);
        LLAMA_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    void seek(size_t offset, int whence) const {
        int ret = std::fseek(fp, (long) offset, whence);
        LLAMA_ASSERT(ret == 0); // same
    }

    void read_raw(void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error(std::string("unexpectedly reached end of file"));
        }
    }

    std::uint32_t read_u32() const {
        std::uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    std::string read_string(std::uint32_t len) const {
        std::vector<char> chars(len);
        read_raw(chars.data(), len);
        return std::string(chars.data(), len);
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(std::uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};

struct llama_mmap {
    void * addr;
    size_t size;

    llama_mmap(const llama_mmap &) = delete;

#ifdef _POSIX_MAPPED_FILES
    static constexpr bool SUPPORTED = true;

    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1 /* -1 = max value */, bool numa = false, bool has_lora = false) {
        size = file->size;
        int fd = fileno(file->fp);
        int flags = (has_lora) ? MAP_PRIVATE : MAP_SHARED;
        // prefetch/readahead impairs performance on NUMA systems
        if (numa) {
            prefetch = 0;
        }
        addr = mmap(NULL, file->size, (has_lora) ? PROT_READ | PROT_WRITE : PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0) {
            // Advise the kernel to preload the mapped memory
            if (madvise(addr, std::min(file->size, prefetch), MADV_WILLNEED)) {
                fprintf(stderr, "warning: madvise(.., MADV_WILLNEED) failed: %s\n",
                        strerror(errno));
            }
        }
        if (numa) {
            // advise the kernel not to use readahead
            // (because the next page might not belong on the same node)
            if (madvise(addr, file->size, MADV_RANDOM)) {
                fprintf(stderr, "warning: madvise(.., MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }
    }

    ~llama_mmap() {
        munmap(addr, size);
    }
#else
    static constexpr bool SUPPORTED = false;

    llama_mmap(struct llama_file *, bool prefetch = true, bool numa = false) {
        (void) prefetch;
        (void) numa;

        throw std::runtime_error(std::string("mmap not supported"));
    }
#endif
};

// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
struct llama_mlock {
    void * addr = NULL;
    size_t size = 0;
    bool failed_already = false;

    llama_mlock() {}
    llama_mlock(const llama_mlock &) = delete;

    ~llama_mlock() {
        if (size) {
            raw_unlock(addr, size);
        }
    }

    void init(void * ptr) {
        LLAMA_ASSERT(addr == NULL && size == 0);
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        LLAMA_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            if (raw_lock((uint8_t *) addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

#ifdef _POSIX_MEMLOCK_RANGE
    static constexpr bool SUPPORTED = true;

    static size_t lock_granularity() {
        return (size_t) sysconf(_SC_PAGESIZE);
    }

#ifdef __APPLE__
#define MLOCK_SUGGESTION                                                                            \
    "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
    "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION \
    "Try increasing RLIMIT_MLOCK ('ulimit -l' as root).\n"
#endif

    bool raw_lock(const void * addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }
        char * errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);

        // Check if the resource limit is fine after all
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit))
            suggest = false;
        if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size))
            suggest = false;

        fprintf(stderr, "warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

#undef MLOCK_SUGGESTION

    static void raw_unlock(void * addr, size_t size) {
        if (munlock(addr, size)) {
            fprintf(stderr, "warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    size_t lock_granularity() {
        return (size_t) 65536;
    }

    bool raw_lock(const void * addr, size_t len) {
        fprintf(stderr, "warning: mlock not supported on this system\n");
        return false;
    }

    void raw_unlock(const void * addr, size_t len) {}
#endif
};

// Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct llama_buffer {
    uint8_t * addr = NULL;
    size_t size = 0;

    llama_buffer() = default;

    void resize(size_t len) {
#ifdef GGML_USE_METAL
        free(addr);
        int result = posix_memalign((void **) &addr, getpagesize(), len);
        if (result == 0) {
            memset(addr, 0, len);
        } else {
            addr = NULL;
        }
#else
        delete[] addr;
        addr = new uint8_t[len];
#endif
        size = len;
    }

    ~llama_buffer() {
#ifdef GGML_USE_METAL
        free(addr);
#else
        delete[] addr;
#endif
        addr = NULL;
    }

    // disable copy and move
    llama_buffer(const llama_buffer &) = delete;
    llama_buffer(llama_buffer &&) = delete;
    llama_buffer & operator=(const llama_buffer &) = delete;
    llama_buffer & operator=(llama_buffer &&) = delete;
};

typedef llama_buffer llama_ctx_buffer;

#endif
