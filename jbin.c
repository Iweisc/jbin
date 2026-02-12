#include "jbin.h"

#if defined(__x86_64__) || defined(_M_X64)
  #ifdef __AVX2__
    #define JBIN_AVX2
    #include <immintrin.h>
    #if defined(__PCLMUL__)
      #define JBIN_TWOPASS
      #include <wmmintrin.h>
    #endif
  #elif defined(__SSE2__)
    #define JBIN_SSE2
    #include <emmintrin.h>
  #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define LIKELY(x)    __builtin_expect(!!(x), 1)
  #define UNLIKELY(x)  __builtin_expect(!!(x), 0)
  #define FINLINE      __attribute__((always_inline)) static inline
  #define NOINLINE     __attribute__((noinline)) static
#else
  #define LIKELY(x)    (x)
  #define UNLIKELY(x)  (x)
  #define FINLINE      static inline
  #define NOINLINE     static
#endif

FINLINE int is_8_digits(const uint8_t *ptr) {
    uint64_t val;
    __builtin_memcpy(&val, ptr, 8);
    return ((val & 0xF0F0F0F0F0F0F0F0ULL) |
            (((val + 0x0606060606060606ULL) & 0xF0F0F0F0F0F0F0F0ULL) >> 4))
           == 0x3333333333333333ULL;
}

FINLINE void skip_digits(const uint8_t **cur, const uint8_t *end) {
    while (*cur + 8 <= end && is_8_digits(*cur)) *cur += 8;
    while (*cur < end && (uint32_t)(**cur - '0') <= 9u) (*cur)++;
}

FINLINE void skip_digits_nobound(const uint8_t **cur) {
#ifdef JBIN_AVX2
    const __m256i v2f = _mm256_set1_epi8(0x2F);
    const __m256i v3a = _mm256_set1_epi8(0x3A);
    {
        __m256i v = _mm256_loadu_si256((const __m256i *)*cur);
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(
            _mm256_and_si256(_mm256_cmpgt_epi8(v, v2f), _mm256_cmpgt_epi8(v3a, v)));
        if (LIKELY(mask != 0xFFFFFFFFu)) {
            *cur += __builtin_ctz(~mask);
            return;
        }
        *cur += 32;
    }
#endif
    while (is_8_digits(*cur)) *cur += 8;
    while ((uint32_t)(**cur - '0') <= 9u) (*cur)++;
}

#define X 0xFF
static const uint8_t HEX[256] = {
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, X, X, X, X, X, X,
    X,10,11,12,13,14,15, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X,10,11,12,13,14,15, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
};
#undef X

static const uint8_t ESC[256] = {
    ['"']  = 0x22, ['\\'] = 0x5C, ['/'] = 0x2F,
    ['b']  = 0x08, ['f']  = 0x0C, ['n'] = 0x0A,
    ['r']  = 0x0D, ['t']  = 0x09,
};

typedef struct {
    const uint8_t *cur;
    const uint8_t *end;
    const uint8_t *input;
    JbinArena     *arena;
    JbinError      error;
    uint32_t       error_pos;
    uint32_t       node_count;
    uint32_t       last_pos;
} P;

#define POS(p) ((uint32_t)((p)->cur - (p)->input))

FINLINE void fail(P *p, JbinError err) {
    if (p->error == JBIN_OK) {
        p->error = err;
        p->error_pos = POS(p);
    }
}

FINLINE uint32_t alloc_node(P *p, JbinType type) {
    uint32_t idx = p->node_count;
    if (UNLIKELY(idx >= JBIN_MAX_NODES)) {
        fail(p, JBIN_ERR_NODES_FULL);
        return JBIN_NONE;
    }
    p->node_count = idx + 1;
    jbin_node_set(&p->arena->nodes[idx], type, JBIN_NONE);
    return idx;
}

FINLINE void skip_ws(P *p) {
    const uint8_t *cur = p->cur;
    const uint8_t *end = p->end;

    if (LIKELY(cur < end && *cur > 0x20)) return;

#ifdef JBIN_AVX2
    const __m256i vspace = _mm256_set1_epi8(' ');
    const __m256i vnewline = _mm256_set1_epi8('\n');
    const __m256i vreturn = _mm256_set1_epi8('\r');
    const __m256i vtab = _mm256_set1_epi8('\t');
    while (cur + 32 <= end) {
        __m256i v = _mm256_loadu_si256((const __m256i *)cur);
        __m256i ws = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_cmpeq_epi8(v, vspace),
                _mm256_cmpeq_epi8(v, vnewline)),
            _mm256_or_si256(
                _mm256_cmpeq_epi8(v, vreturn),
                _mm256_cmpeq_epi8(v, vtab)));
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(ws);
        if (mask != 0xFFFFFFFFu) {
            p->cur = cur + (uint32_t)__builtin_ctz(~mask);
            return;
        }
        cur += 32;
    }
#elif defined(JBIN_SSE2)
    while (cur + 16 <= end) {
        __m128i v = _mm_loadu_si128((const __m128i *)cur);
        __m128i ws = _mm_or_si128(
            _mm_or_si128(
                _mm_cmpeq_epi8(v, _mm_set1_epi8(' ')),
                _mm_cmpeq_epi8(v, _mm_set1_epi8('\n'))),
            _mm_or_si128(
                _mm_cmpeq_epi8(v, _mm_set1_epi8('\r')),
                _mm_cmpeq_epi8(v, _mm_set1_epi8('\t'))));
        int mask = _mm_movemask_epi8(ws);
        if (mask != 0xFFFF) {
            p->cur = cur + (uint32_t)__builtin_ctz(~mask);
            return;
        }
        cur += 16;
    }
#endif

    while (cur < end) {
        uint8_t c = *cur;
        if (c != 0x20 && c != 0x0A && c != 0x0D && c != 0x09) break;
        cur++;
    }
    p->cur = cur;
}

static const uint8_t *scan_string_end(const uint8_t *cur, const uint8_t *end,
                                       int *all_ascii) {
    int ascii = 1;

#ifdef JBIN_AVX2
    {
        const __m256i vq  = _mm256_set1_epi8('"');
        const __m256i vbs = _mm256_set1_epi8('\\');
        const __m256i vxr = _mm256_set1_epi8((char)0x80);
        const __m256i vth = _mm256_set1_epi8((char)0xA0);

        while (cur + 32 <= end) {
            __m256i v = _mm256_loadu_si256((const __m256i *)cur);
            uint32_t hi = (uint32_t)_mm256_movemask_epi8(v);
            __m256i hit = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_cmpeq_epi8(v, vq),
                    _mm256_cmpeq_epi8(v, vbs)),
                _mm256_cmpgt_epi8(vth, _mm256_xor_si256(v, vxr)));
            uint32_t mask = (uint32_t)_mm256_movemask_epi8(hit);
            if (mask) {
                uint32_t hit_pos = (uint32_t)__builtin_ctz(mask);
                if (hi & ((1u << hit_pos) - 1)) ascii = 0;
                *all_ascii = ascii;
                return cur + hit_pos;
            }
            if (hi) ascii = 0;
            cur += 32;
        }
    }
#elif defined(JBIN_SSE2)
    {
        const __m128i vq  = _mm_set1_epi8('"');
        const __m128i vbs = _mm_set1_epi8('\\');
        const __m128i vxr = _mm_set1_epi8((char)0x80);
        const __m128i vth = _mm_set1_epi8((char)0xA0);

        while (cur + 16 <= end) {
            __m128i v = _mm_loadu_si128((const __m128i *)cur);
            uint32_t hi = (uint32_t)_mm_movemask_epi8(v);
            __m128i hit = _mm_or_si128(
                _mm_or_si128(
                    _mm_cmpeq_epi8(v, vq),
                    _mm_cmpeq_epi8(v, vbs)),
                _mm_cmpgt_epi8(vth, _mm_xor_si128(v, vxr)));
            uint32_t mask = (uint32_t)_mm_movemask_epi8(hit);
            if (mask) {
                uint32_t hit_pos = (uint32_t)__builtin_ctz(mask);
                if (hi & ((1u << hit_pos) - 1)) ascii = 0;
                *all_ascii = ascii;
                return cur + hit_pos;
            }
            if (hi) ascii = 0;
            cur += 16;
        }
    }
#endif

    while (cur < end) {
        uint8_t c = *cur;
        if (c >= 0x80) ascii = 0;
        if (c == '"' || c == '\\' || c < 0x20) {
            *all_ascii = ascii;
            return cur;
        }
        cur++;
    }
    *all_ascii = ascii;
    return cur;
}

FINLINE int arena_bulk(P *p, const uint8_t *src, uint32_t n) {
    if (UNLIKELY(p->arena->string_used + n + 1 > JBIN_MAX_STRING)) {
        fail(p, JBIN_ERR_STRING_FULL);
        return 0;
    }
    uint8_t *dst = (uint8_t *)p->arena->strings + p->arena->string_used;
    uint32_t i = 0;
#ifdef JBIN_AVX2
    while (i + 32 <= n) {
        _mm256_storeu_si256((__m256i *)(dst + i),
                            _mm256_loadu_si256((const __m256i *)(src + i)));
        i += 32;
    }
#elif defined(JBIN_SSE2)
    while (i + 16 <= n) {
        _mm_storeu_si128((__m128i *)(dst + i),
                         _mm_loadu_si128((const __m128i *)(src + i)));
        i += 16;
    }
#endif
    while (i + 8 <= n) {
        __builtin_memcpy(dst + i, src + i, 8);
        i += 8;
    }
    while (i < n) { dst[i] = src[i]; i++; }
    p->arena->string_used += n;
    return 1;
}

FINLINE const uint8_t *scan_copy_string(P *p, const uint8_t *cur) {
    uint8_t *dst = (uint8_t *)p->arena->strings + p->arena->string_used;
    uint32_t capacity = JBIN_MAX_STRING - p->arena->string_used;
    uint32_t n = 0;

#ifdef JBIN_AVX2
    {
        const __m256i vq  = _mm256_set1_epi8('"');
        const __m256i vbs = _mm256_set1_epi8('\\');
        const __m256i vxr = _mm256_set1_epi8((char)0x80);
        const __m256i vth = _mm256_set1_epi8((char)0xA0);

        while (cur + 32 <= p->end && n + 32 <= capacity) {
            __m256i v = _mm256_loadu_si256((const __m256i *)cur);
            uint32_t qmask = (uint32_t)_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(v, vq));
            uint32_t bmask = (uint32_t)_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(v, vbs));
            if (LIKELY(!(qmask | bmask))) {
                _mm256_storeu_si256((__m256i *)(dst + n), v);
                n += 32;
                cur += 32;
                continue;
            }
            uint32_t cmask = (uint32_t)_mm256_movemask_epi8(
                _mm256_cmpgt_epi8(vth, _mm256_xor_si256(v, vxr)));
            uint32_t pos = (uint32_t)__builtin_ctz(qmask | bmask | cmask);
            _mm256_storeu_si256((__m256i *)(dst + n), v);
            p->arena->string_used += n + pos;
            return cur + pos;
        }
    }
#elif defined(JBIN_SSE2)
    {
        const __m128i vq  = _mm_set1_epi8('"');
        const __m128i vbs = _mm_set1_epi8('\\');
        const __m128i vxr = _mm_set1_epi8((char)0x80);
        const __m128i vth = _mm_set1_epi8((char)0xA0);

        while (cur + 16 <= p->end && n + 16 <= capacity) {
            __m128i v = _mm_loadu_si128((const __m128i *)cur);
            uint32_t qmask = (uint32_t)_mm_movemask_epi8(
                _mm_cmpeq_epi8(v, vq));
            uint32_t bmask = (uint32_t)_mm_movemask_epi8(
                _mm_cmpeq_epi8(v, vbs));
            if (LIKELY(!(qmask | bmask))) {
                _mm_storeu_si128((__m128i *)(dst + n), v);
                n += 16;
                cur += 16;
                continue;
            }
            uint32_t cmask = (uint32_t)_mm_movemask_epi8(
                _mm_cmpgt_epi8(vth, _mm_xor_si128(v, vxr)));
            uint32_t pos = (uint32_t)__builtin_ctz(qmask | bmask | cmask);
            _mm_storeu_si128((__m128i *)(dst + n), v);
            p->arena->string_used += n + pos;
            return cur + pos;
        }
    }
#endif

    while (cur < p->end && n < capacity) {
        uint8_t c = *cur;
        if (c == '"' || c == '\\' || c < 0x20) break;
        dst[n++] = c;
        cur++;
    }
    p->arena->string_used += n;

    if (UNLIKELY(n >= capacity && cur < p->end)) {
        uint8_t c = *cur;
        if (c != '"' && c != '\\' && c >= 0x20) {
            fail(p, JBIN_ERR_STRING_FULL);
        }
    }

    return cur;
}

FINLINE int str_push(P *p, char c) {
    if (UNLIKELY(p->arena->string_used >= JBIN_MAX_STRING)) {
        fail(p, JBIN_ERR_STRING_FULL);
        return 0;
    }
    p->arena->strings[p->arena->string_used++] = c;
    return 1;
}

NOINLINE int validate_utf8(const uint8_t *data, uint32_t len) {
    uint32_t i = 0;

    while (i < len) {
        uint8_t b = data[i];
        if (b < 0x80) { i++; continue; }

        uint32_t cp;
        uint32_t n;

        if (b >= 0xC2 && b <= 0xDF)      { cp = b & 0x1Fu; n = 2; }
        else if (b >= 0xE0 && b <= 0xEF)  { cp = b & 0x0Fu; n = 3; }
        else if (b >= 0xF0 && b <= 0xF4)  { cp = b & 0x07u; n = 4; }
        else return 0;

        if (i + n > len) return 0;

        for (uint32_t j = 1; j < n; j++) {
            if ((data[i + j] & 0xC0) != 0x80) return 0;
            cp = (cp << 6) | (data[i + j] & 0x3Fu);
        }

        if (n == 3 && cp < 0x800)        return 0;
        if (n == 4 && cp < 0x10000)      return 0;
        if (cp >= 0xD800 && cp <= 0xDFFF) return 0;
        if (cp > 0x10FFFF)               return 0;

        i += n;
    }
    return 1;
}

FINLINE int encode_utf8(P *p, uint32_t cp) {
    if (cp <= 0x7F)
        return str_push(p, (char)cp);
    if (cp <= 0x7FF)
        return str_push(p, (char)(0xC0 | (cp >> 6)))
            && str_push(p, (char)(0x80 | (cp & 0x3F)));
    if (cp <= 0xFFFF)
        return str_push(p, (char)(0xE0 | (cp >> 12)))
            && str_push(p, (char)(0x80 | ((cp >> 6) & 0x3F)))
            && str_push(p, (char)(0x80 | (cp & 0x3F)));
    if (cp <= 0x10FFFF)
        return str_push(p, (char)(0xF0 | (cp >> 18)))
            && str_push(p, (char)(0x80 | ((cp >> 12) & 0x3F)))
            && str_push(p, (char)(0x80 | ((cp >> 6) & 0x3F)))
            && str_push(p, (char)(0x80 | (cp & 0x3F)));
    return 0;
}

FINLINE int parse_4hex(P *p, uint32_t *out) {
    if (UNLIKELY(p->cur + 4 > p->end)) return 0;
    uint8_t a = HEX[p->cur[0]], b = HEX[p->cur[1]];
    uint8_t c = HEX[p->cur[2]], d = HEX[p->cur[3]];
    if (UNLIKELY((a | b | c | d) & 0x80)) return 0;
    *out = ((uint32_t)a << 12) | ((uint32_t)b << 8) | ((uint32_t)c << 4) | d;
    p->cur += 4;
    return 1;
}

#ifdef JBIN_AVX2
NOINLINE uint32_t parse_string_fused(P *p) {
    if (UNLIKELY(p->cur >= p->end || *p->cur != '"')) {
        fail(p, JBIN_ERR_UNEXPECTED);
        return JBIN_NONE;
    }
    p->cur++;

    uint32_t node = alloc_node(p, JBIN_STRING);
    if (UNLIKELY(node == JBIN_NONE)) return JBIN_NONE;

    uint32_t arena_start = p->arena->string_used;
    uint8_t *dst_base = (uint8_t *)p->arena->strings + arena_start;
    uint8_t *dst = dst_base;
    uint32_t capacity = JBIN_MAX_STRING - arena_start;
    const uint8_t *src = p->cur;
    const uint8_t *end = p->end;
    uint8_t *dst_limit = dst_base + capacity - 31;
    int has_non_ascii = 0;

    const __m256i vquote = _mm256_set1_epi8('"');
    const __m256i vbackslash = _mm256_set1_epi8('\\');
    const __m256i vxor = _mm256_set1_epi8((char)0x80);
    const __m256i vthresh = _mm256_set1_epi8((char)0xA0);

    while (src + 32 <= end && dst <= dst_limit) {
        __m256i v = _mm256_loadu_si256((const __m256i *)src);
        uint32_t qmask = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vquote));
        uint32_t bmask = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vbackslash));

        if (LIKELY(!(qmask | bmask))) {
            _mm256_storeu_si256((__m256i *)dst, v);
            has_non_ascii |= (int)_mm256_movemask_epi8(v);
            src += 32;
            dst += 32;
            continue;
        }

        uint32_t cmask = (uint32_t)_mm256_movemask_epi8(
            _mm256_cmpgt_epi8(vthresh, _mm256_xor_si256(v, vxor)));
        uint32_t pos = (uint32_t)__builtin_ctz(qmask | bmask | cmask);

        has_non_ascii |= (int)_mm256_movemask_epi8(v);
        _mm256_storeu_si256((__m256i *)dst, v);
        dst += pos;
        src += pos;

        if (UNLIKELY(cmask & (1u << pos))) {
            p->cur = src;
            fail(p, JBIN_ERR_CONTROL_CHAR);
            return JBIN_NONE;
        }

        if (qmask & (1u << pos)) {
            uint32_t slen = (uint32_t)(dst - dst_base);
            p->arena->string_used = arena_start + slen;
            if (UNLIKELY(has_non_ascii && !validate_utf8(dst_base, slen))) {
                p->cur = src;
                fail(p, JBIN_ERR_BAD_UTF8);
                return JBIN_NONE;
            }
            p->arena->nodes[node].str_off = arena_start;
            p->arena->nodes[node].str_len = slen;
            p->cur = src + 1;
            return node;
        }

        if (UNLIKELY(src + 1 >= end)) {
            p->cur = src;
            fail(p, JBIN_ERR_UNTERMINATED_STRING);
            return JBIN_NONE;
        }
        uint8_t esc_ch = src[1];
        src += 2;
        uint8_t rep = ESC[esc_ch];
        if (LIKELY(rep)) {
            *dst++ = rep;
            while (src + 1 < end && *src == '\\') {
                uint8_t nr = ESC[src[1]];
                if (!nr) break;
                *dst++ = nr;
                src += 2;
            }
            continue;
        }
        if (LIKELY(esc_ch == 'u')) {
                if (UNLIKELY(src + 4 > end)) {
                    p->cur = src - 2;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return JBIN_NONE;
                }
                uint8_t a = HEX[src[0]], b = HEX[src[1]];
                uint8_t c = HEX[src[2]], d = HEX[src[3]];
                if (UNLIKELY((a | b | c | d) & 0x80)) {
                    p->cur = src - 2;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return JBIN_NONE;
                }
                uint32_t cp = ((uint32_t)a << 12) | ((uint32_t)b << 8)
                            | ((uint32_t)c << 4) | d;
                src += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (UNLIKELY(src + 6 > end || src[0] != '\\' || src[1] != 'u')) {
                        p->cur = src - 6;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return JBIN_NONE;
                    }
                    src += 2;
                    uint8_t la = HEX[src[0]], lb = HEX[src[1]];
                    uint8_t lc = HEX[src[2]], ld = HEX[src[3]];
                    if (UNLIKELY((la | lb | lc | ld) & 0x80)) {
                        p->cur = src - 2;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return JBIN_NONE;
                    }
                    uint32_t low = ((uint32_t)la << 12) | ((uint32_t)lb << 8)
                                 | ((uint32_t)lc << 4) | ld;
                    src += 4;
                    if (UNLIKELY(low < 0xDC00 || low > 0xDFFF)) {
                        p->cur = src - 6;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return JBIN_NONE;
                    }
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                } else if (UNLIKELY(cp >= 0xDC00 && cp <= 0xDFFF)) {
                    p->cur = src - 6;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return JBIN_NONE;
                }
                if (UNLIKELY((uint32_t)(dst - dst_base) + 4 > capacity)) {
                    p->cur = src;
                    fail(p, JBIN_ERR_STRING_FULL);
                    return JBIN_NONE;
                }
                if (cp <= 0x7F) {
                    *dst++ = (uint8_t)cp;
                } else if (cp <= 0x7FF) {
                    *dst++ = (uint8_t)(0xC0 | (cp >> 6));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                } else if (cp <= 0xFFFF) {
                    *dst++ = (uint8_t)(0xE0 | (cp >> 12));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                } else {
                    *dst++ = (uint8_t)(0xF0 | (cp >> 18));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                }
        } else {
            p->cur = src - 2;
            fail(p, JBIN_ERR_BAD_ESCAPE);
            return JBIN_NONE;
        }
    }

    while (src < end) {
        if (UNLIKELY((uint32_t)(dst - dst_base) >= capacity)) {
            p->cur = src;
            fail(p, JBIN_ERR_STRING_FULL);
            return JBIN_NONE;
        }

        uint8_t ch = *src;

        if (ch == '"') {
            uint32_t slen = (uint32_t)(dst - dst_base);
            p->arena->string_used = arena_start + slen;
            if (UNLIKELY(has_non_ascii && !validate_utf8(dst_base, slen))) {
                p->cur = src;
                fail(p, JBIN_ERR_BAD_UTF8);
                return JBIN_NONE;
            }
            p->arena->nodes[node].str_off = arena_start;
            p->arena->nodes[node].str_len = slen;
            p->cur = src + 1;
            return node;
        }

        if (UNLIKELY(ch < 0x20)) {
            p->cur = src;
            fail(p, JBIN_ERR_CONTROL_CHAR);
            return JBIN_NONE;
        }

        if (ch == '\\') {
            if (UNLIKELY(src + 1 >= end)) {
                p->cur = src;
                fail(p, JBIN_ERR_UNTERMINATED_STRING);
                return JBIN_NONE;
            }
            uint8_t esc_ch = src[1];
            src += 2;
            uint8_t rep = ESC[esc_ch];
            if (LIKELY(rep)) {
                *dst++ = rep;
            } else if (LIKELY(esc_ch == 'u')) {
                if (UNLIKELY(src + 4 > end)) {
                    p->cur = src - 2;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return JBIN_NONE;
                }
                uint8_t a = HEX[src[0]], b = HEX[src[1]];
                uint8_t c = HEX[src[2]], d = HEX[src[3]];
                if (UNLIKELY((a | b | c | d) & 0x80)) {
                    p->cur = src - 2;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return JBIN_NONE;
                }
                uint32_t cp = ((uint32_t)a << 12) | ((uint32_t)b << 8)
                            | ((uint32_t)c << 4) | d;
                src += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (UNLIKELY(src + 6 > end || src[0] != '\\' || src[1] != 'u')) {
                        p->cur = src - 6;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return JBIN_NONE;
                    }
                    src += 2;
                    uint8_t la = HEX[src[0]], lb = HEX[src[1]];
                    uint8_t lc = HEX[src[2]], ld = HEX[src[3]];
                    if (UNLIKELY((la | lb | lc | ld) & 0x80)) {
                        p->cur = src - 2;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return JBIN_NONE;
                    }
                    uint32_t low = ((uint32_t)la << 12) | ((uint32_t)lb << 8)
                                 | ((uint32_t)lc << 4) | ld;
                    src += 4;
                    if (UNLIKELY(low < 0xDC00 || low > 0xDFFF)) {
                        p->cur = src - 6;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return JBIN_NONE;
                    }
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                } else if (UNLIKELY(cp >= 0xDC00 && cp <= 0xDFFF)) {
                    p->cur = src - 6;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return JBIN_NONE;
                }
                if (UNLIKELY((uint32_t)(dst - dst_base) + 4 > capacity)) {
                    p->cur = src;
                    fail(p, JBIN_ERR_STRING_FULL);
                    return JBIN_NONE;
                }
                if (cp <= 0x7F) {
                    *dst++ = (uint8_t)cp;
                } else if (cp <= 0x7FF) {
                    *dst++ = (uint8_t)(0xC0 | (cp >> 6));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                } else if (cp <= 0xFFFF) {
                    *dst++ = (uint8_t)(0xE0 | (cp >> 12));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                } else {
                    *dst++ = (uint8_t)(0xF0 | (cp >> 18));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                }
            } else {
                p->cur = src - 2;
                fail(p, JBIN_ERR_BAD_ESCAPE);
                return JBIN_NONE;
            }
            continue;
        }

        has_non_ascii |= (ch >> 7);
        *dst++ = ch;
        src++;
    }

    p->cur = src;
    fail(p, JBIN_ERR_UNTERMINATED_STRING);
    return JBIN_NONE;
}
#else
NOINLINE uint32_t parse_string_fused(P *p) {
    if (UNLIKELY(p->cur >= p->end || *p->cur != '"')) {
        fail(p, JBIN_ERR_UNEXPECTED);
        return JBIN_NONE;
    }

    const uint8_t *src = p->cur + 1;
    const uint8_t *end = p->end;
    uint32_t arena_start = p->arena->string_used;
    uint8_t *dst_base = (uint8_t *)p->arena->strings + arena_start;
    uint8_t *dst = dst_base;
    uint8_t *dst_end = (uint8_t *)p->arena->strings + JBIN_MAX_STRING;

    uint32_t node = alloc_node(p, JBIN_STRING);
    if (UNLIKELY(node == JBIN_NONE)) return JBIN_NONE;

    while (LIKELY(src < end)) {
        uint8_t c = *src;

        if (LIKELY(c >= 0x20 && c != '"' && c != '\\')) {
            if (UNLIKELY(dst >= dst_end)) {
                p->cur = src;
                fail(p, JBIN_ERR_STRING_FULL);
                return JBIN_NONE;
            }
            *dst++ = c;
            src++;
            continue;
        }

        if (c == '"') {
            uint32_t slen = (uint32_t)(dst - dst_base);
            p->arena->string_used = arena_start + slen;
            if (UNLIKELY(!validate_utf8(dst_base, slen))) {
                p->cur = src;
                fail(p, JBIN_ERR_BAD_UTF8);
                return JBIN_NONE;
            }
            p->arena->nodes[node].str_off = arena_start;
            p->arena->nodes[node].str_len = slen;
            p->cur = src + 1;
            return node;
        }

        if (c == '\\') {
            src++;
            if (UNLIKELY(src >= end)) {
                p->cur = src;
                fail(p, JBIN_ERR_UNTERMINATED_STRING);
                return JBIN_NONE;
            }
            uint8_t esc_ch = *src++;
            uint8_t rep = ESC[esc_ch];
            if (LIKELY(rep)) {
                if (UNLIKELY(dst >= dst_end)) goto full;
                *dst++ = rep;
                while (src < end && *src == '\\' && src + 1 < end) {
                    uint8_t nr = ESC[src[1]];
                    if (!nr) break;
                    if (UNLIKELY(dst >= dst_end)) goto full;
                    *dst++ = nr;
                    src += 2;
                }
            } else if (LIKELY(esc_ch == 'u')) {
                if (UNLIKELY(src + 4 > end)) {
                    p->cur = src;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return JBIN_NONE;
                }
                uint8_t a = HEX[src[0]], b = HEX[src[1]];
                uint8_t hc = HEX[src[2]], d = HEX[src[3]];
                if (UNLIKELY((a | b | hc | d) & 0x80)) {
                    p->cur = src;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return JBIN_NONE;
                }
                uint32_t cp = ((uint32_t)a << 12) | ((uint32_t)b << 8) | ((uint32_t)hc << 4) | d;
                src += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (UNLIKELY(src + 6 > end || src[0] != '\\' || src[1] != 'u')) {
                        p->cur = src;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return JBIN_NONE;
                    }
                    src += 2;
                    uint8_t la = HEX[src[0]], lb = HEX[src[1]];
                    uint8_t lc = HEX[src[2]], ld = HEX[src[3]];
                    if (UNLIKELY((la | lb | lc | ld) & 0x80)) {
                        p->cur = src;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return JBIN_NONE;
                    }
                    uint32_t low = ((uint32_t)la << 12) | ((uint32_t)lb << 8) | ((uint32_t)lc << 4) | ld;
                    src += 4;
                    if (UNLIKELY(low < 0xDC00 || low > 0xDFFF)) {
                        p->cur = src;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return JBIN_NONE;
                    }
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                } else if (UNLIKELY(cp >= 0xDC00 && cp <= 0xDFFF)) {
                    p->cur = src;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return JBIN_NONE;
                }
                if (cp <= 0x7F) {
                    if (UNLIKELY(dst >= dst_end)) goto full;
                    *dst++ = (uint8_t)cp;
                } else if (cp <= 0x7FF) {
                    if (UNLIKELY(dst + 2 > dst_end)) goto full;
                    *dst++ = (uint8_t)(0xC0 | (cp >> 6));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                } else if (cp <= 0xFFFF) {
                    if (UNLIKELY(dst + 3 > dst_end)) goto full;
                    *dst++ = (uint8_t)(0xE0 | (cp >> 12));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                } else {
                    if (UNLIKELY(dst + 4 > dst_end)) goto full;
                    *dst++ = (uint8_t)(0xF0 | (cp >> 18));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                }
            } else {
                p->cur = src - 1;
                fail(p, JBIN_ERR_BAD_ESCAPE);
                return JBIN_NONE;
            }
            continue;
        }

        p->cur = src;
        fail(p, JBIN_ERR_CONTROL_CHAR);
        return JBIN_NONE;
    }

    p->cur = src;
    fail(p, JBIN_ERR_UNTERMINATED_STRING);
    return JBIN_NONE;

full:
    p->cur = src;
    fail(p, JBIN_ERR_STRING_FULL);
    return JBIN_NONE;
}
#endif

static uint32_t parse_number(P *p) {
    const uint8_t *start = p->cur;

    if (p->cur < p->end && *p->cur == '-') p->cur++;

    if (UNLIKELY(p->cur >= p->end || (uint32_t)(*p->cur - '0') > 9u)) {
        fail(p, JBIN_ERR_BAD_NUMBER);
        return JBIN_NONE;
    }

    if (*p->cur == '0') {
        p->cur++;
        if (UNLIKELY(p->cur < p->end && (uint32_t)(*p->cur - '0') <= 9u)) {
            fail(p, JBIN_ERR_BAD_NUMBER);
            return JBIN_NONE;
        }
    } else {
        p->cur++;
        skip_digits(&p->cur, p->end);
    }

    if (p->cur < p->end && *p->cur == '.') {
        p->cur++;
        if (UNLIKELY(p->cur >= p->end || (uint32_t)(*p->cur - '0') > 9u)) {
            fail(p, JBIN_ERR_BAD_NUMBER);
            return JBIN_NONE;
        }
        p->cur++;
        skip_digits(&p->cur, p->end);
    }

    if (p->cur < p->end && (*p->cur == 'e' || *p->cur == 'E')) {
        p->cur++;
        if (p->cur < p->end && (*p->cur == '+' || *p->cur == '-'))
            p->cur++;
        if (UNLIKELY(p->cur >= p->end || (uint32_t)(*p->cur - '0') > 9u)) {
            fail(p, JBIN_ERR_BAD_NUMBER);
            return JBIN_NONE;
        }
        do { p->cur++; } while (p->cur < p->end && (uint32_t)(*p->cur - '0') <= 9u);
    }

    uint32_t node = alloc_node(p, JBIN_NUMBER);
    if (UNLIKELY(node == JBIN_NONE)) return JBIN_NONE;

    p->arena->nodes[node].str_off = (uint32_t)(start - p->input) | JBIN_INPUT_REF;
    p->arena->nodes[node].str_len = (uint32_t)(p->cur - start);

    return node;
}

#ifdef JBIN_TWOPASS
FINLINE uint32_t parse_number_s2(P *p, const uint8_t *start,
                                 JbinNode *nodes, uint32_t nc) {
    const uint8_t *cur = start;

    if (*cur == '-') cur++;

    if (UNLIKELY((uint32_t)(*cur - '0') > 9u)) {
        p->cur = cur;
        fail(p, JBIN_ERR_BAD_NUMBER);
        return JBIN_NONE;
    }

    if (*cur == '0') {
        cur++;
        if (UNLIKELY((uint32_t)(*cur - '0') <= 9u)) {
            p->cur = cur;
            fail(p, JBIN_ERR_BAD_NUMBER);
            return JBIN_NONE;
        }
    } else {
        cur++;
        skip_digits_nobound(&cur);
    }

    if (*cur == '.') {
        cur++;
        if (UNLIKELY((uint32_t)(*cur - '0') > 9u)) {
            p->cur = cur;
            fail(p, JBIN_ERR_BAD_NUMBER);
            return JBIN_NONE;
        }
        cur++;
        skip_digits_nobound(&cur);
    }

    if (*cur == 'e' || *cur == 'E') {
        cur++;
        if (*cur == '+' || *cur == '-') cur++;
        if (UNLIKELY((uint32_t)(*cur - '0') > 9u)) {
            p->cur = cur;
            fail(p, JBIN_ERR_BAD_NUMBER);
            return JBIN_NONE;
        }
        do { cur++; } while ((uint32_t)(*cur - '0') <= 9u);
    }

    if (UNLIKELY(*cur > 0x20 && *cur != ',' && *cur != ']' && *cur != '}')) {
        p->cur = cur;
        fail(p, JBIN_ERR_BAD_NUMBER);
        return JBIN_NONE;
    }

    if (UNLIKELY(nc >= JBIN_MAX_NODES)) {
        fail(p, JBIN_ERR_NODES_FULL);
        return JBIN_NONE;
    }

    jbin_node_set(&nodes[nc], JBIN_NUMBER, JBIN_NONE);
    nodes[nc].str_off = (uint32_t)(start - p->input) | JBIN_INPUT_REF;
    nodes[nc].str_len = (uint32_t)(cur - start);

    return nc;
}
#endif

FINLINE uint32_t parse_string_fast(P *p) {
    const uint8_t *str_begin = p->cur + 1;

#ifdef JBIN_AVX2
    if (LIKELY(str_begin + 32 <= p->end)) {
        __m256i v = _mm256_loadu_si256((const __m256i *)str_begin);
        uint32_t qm = (uint32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(v, _mm256_set1_epi8('"')));
        uint32_t bm = (uint32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(v, _mm256_set1_epi8('\\')));
        if (LIKELY(qm)) {
            uint32_t qp = (uint32_t)__builtin_ctz(qm);
            uint32_t before = (1u << qp) - 1;
            uint32_t cm = (uint32_t)_mm256_movemask_epi8(
                _mm256_cmpgt_epi8(
                    _mm256_set1_epi8((char)0xA0),
                    _mm256_xor_si256(v, _mm256_set1_epi8((char)0x80))));
            if (LIKELY(!((bm | cm) & before))) {
                uint32_t hi = (uint32_t)_mm256_movemask_epi8(v);
                if (UNLIKELY(hi & before)) {
                    if (!validate_utf8(str_begin, qp)) {
                        fail(p, JBIN_ERR_BAD_UTF8);
                        return JBIN_NONE;
                    }
                }
                uint32_t node = alloc_node(p, JBIN_STRING);
                if (UNLIKELY(node == JBIN_NONE)) return JBIN_NONE;
                p->arena->nodes[node].str_off = (uint32_t)(str_begin - p->input) | JBIN_INPUT_REF;
                p->arena->nodes[node].str_len = qp;
                p->cur = str_begin + qp + 1;
                return node;
            }
            return parse_string_fused(p);
        }
        if (bm)
            return parse_string_fused(p);
        /* Try second 32-byte chunk for strings 33-64 bytes */
        if (LIKELY(str_begin + 64 <= p->end)) {
            __m256i v2 = _mm256_loadu_si256((const __m256i *)(str_begin + 32));
            uint32_t qm2 = (uint32_t)_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(v2, _mm256_set1_epi8('"')));
            uint32_t bm2 = (uint32_t)_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(v2, _mm256_set1_epi8('\\')));
            if (LIKELY(qm2 && !bm2)) {
                uint32_t qp2 = (uint32_t)__builtin_ctz(qm2);
                uint32_t total_len = 32 + qp2;
                uint32_t cm2 = (uint32_t)_mm256_movemask_epi8(
                    _mm256_cmpgt_epi8(
                        _mm256_set1_epi8((char)0xA0),
                        _mm256_xor_si256(v2, _mm256_set1_epi8((char)0x80))));
                uint32_t before2 = (1u << qp2) - 1;
                if (LIKELY(!(cm2 & before2))) {
                    uint32_t cm1 = (uint32_t)_mm256_movemask_epi8(
                        _mm256_cmpgt_epi8(
                            _mm256_set1_epi8((char)0xA0),
                            _mm256_xor_si256(v, _mm256_set1_epi8((char)0x80))));
                    if (UNLIKELY(cm1)) return parse_string_fused(p);
                    uint32_t hi1 = (uint32_t)_mm256_movemask_epi8(v);
                    uint32_t hi2 = (uint32_t)_mm256_movemask_epi8(v2);
                    if (UNLIKELY((hi1 | (hi2 & before2)))) {
                        if (!validate_utf8(str_begin, total_len)) {
                            fail(p, JBIN_ERR_BAD_UTF8);
                            return JBIN_NONE;
                        }
                    }
                    uint32_t node = alloc_node(p, JBIN_STRING);
                    if (UNLIKELY(node == JBIN_NONE)) return JBIN_NONE;
                    p->arena->nodes[node].str_off = (uint32_t)(str_begin - p->input) | JBIN_INPUT_REF;
                    p->arena->nodes[node].str_len = total_len;
                    p->cur = str_begin + total_len + 1;
                    return node;
                }
            }
            if (bm2) return parse_string_fused(p);
        }
    }
#endif

    int ascii;
    const uint8_t *scan_end = scan_string_end(str_begin, p->end, &ascii);

    if (LIKELY(scan_end < p->end && *scan_end == '"')) {
        uint32_t slen = (uint32_t)(scan_end - str_begin);
        if (UNLIKELY(!ascii && !validate_utf8(str_begin, slen))) {
            fail(p, JBIN_ERR_BAD_UTF8);
            return JBIN_NONE;
        }
        uint32_t node = alloc_node(p, JBIN_STRING);
        if (UNLIKELY(node == JBIN_NONE)) return JBIN_NONE;
        p->arena->nodes[node].str_off = (uint32_t)(str_begin - p->input) | JBIN_INPUT_REF;
        p->arena->nodes[node].str_len = slen;
        p->cur = scan_end + 1;
        return node;
    }

    return parse_string_fused(p);
}

typedef struct {
    uint32_t  node;
    uint32_t *link;
    uint32_t  key;
    uint8_t   is_obj;
} Frame;

#define LINK_SET(lnk, val) \
    (*(lnk) = (*(lnk) & ~JBIN_NEXT_MASK) | (val))

#ifdef JBIN_TWOPASS

FINLINE uint64_t prefix_xor(uint64_t x) {
    __m128i v = _mm_set_epi64x(0, (long long)x);
    v = _mm_clmulepi64_si128(v, _mm_set1_epi8((char)0xFF), 0x00);
    return (uint64_t)_mm_cvtsi128_si64(v);
}

FINLINE uint64_t find_escaped(uint64_t bs, uint64_t *prev_odd_carry) {
    uint64_t escaped = 0;
    uint64_t odd_carry = *prev_odd_carry;

    if (LIKELY(!bs && !odd_carry)) {
        *prev_odd_carry = 0;
        return 0;
    }

    if (odd_carry) {
        if (bs & 1) {
            uint64_t not_bs = ~bs;
            uint32_t cont = not_bs ? (uint32_t)__builtin_ctzll(not_bs) : 64;
            uint64_t run_mask = cont >= 64 ? ~0ULL : (1ULL << cont) - 1;
            bs &= ~run_mask;
            if (!(cont & 1)) {
                if (cont < 64) escaped |= (1ULL << cont);
                else { *prev_odd_carry = 1; return escaped; }
            }
        } else {
            escaped = 1;
        }
    }

    while (bs) {
        uint32_t first = (uint32_t)__builtin_ctzll(bs);
        uint64_t shifted = bs >> first;
        uint64_t not_shifted = ~shifted;
        uint32_t count = not_shifted ? (uint32_t)__builtin_ctzll(not_shifted) : (64 - first);
        uint64_t run_mask = count >= (64 - first)
            ? ~((1ULL << first) - 1)
            : ((1ULL << count) - 1) << first;
        bs &= ~run_mask;
        if (count & 1) {
            uint32_t esc_pos = first + count;
            if (esc_pos < 64) escaped |= (1ULL << esc_pos);
            else { *prev_odd_carry = 1; return escaped; }
        }
    }

    *prev_odd_carry = 0;
    return escaped;
}


typedef struct {
    const uint64_t *bm_cur;
    const uint64_t *bm_end;
    uint32_t block_base;
    uint32_t cur_pos;
    uint64_t mask;
} SiIter;

FINLINE void si_init(SiIter *it, const uint64_t *bm, uint32_t num_blocks) {
    it->bm_cur = bm;
    it->bm_end = bm + num_blocks;
    it->block_base = 0;
    it->mask = num_blocks > 0 ? bm[0] : 0;
    while (!it->mask && it->bm_cur + 1 < it->bm_end) {
        it->bm_cur++;
        it->block_base += 64;
        it->mask = *it->bm_cur;
    }
    it->cur_pos = it->block_base + (uint32_t)__builtin_ctzll(it->mask);
}

FINLINE uint32_t si_pos(const SiIter *it) {
    return it->cur_pos;
}

FINLINE int si_has(const SiIter *it) {
    return it->mask != 0;
}

FINLINE void si_advance(SiIter *it) {
    uint64_t m = _blsr_u64(it->mask);
    if (LIKELY(m)) {
        it->mask = m;
        it->cur_pos = it->block_base + (uint32_t)__builtin_ctzll(m);
        return;
    }
    it->mask = 0;
    while (it->bm_cur + 1 < it->bm_end) {
        it->bm_cur++;
        it->block_base += 64;
        m = *it->bm_cur;
        if (m) {
            it->mask = m;
            it->cur_pos = it->block_base + (uint32_t)__builtin_ctzll(m);
            return;
        }
    }
}

FINLINE uint32_t si_next(SiIter *it) {
    uint32_t p = it->cur_pos;
    uint64_t m = _blsr_u64(it->mask);
    if (LIKELY(m)) {
        it->mask = m;
        it->cur_pos = it->block_base + (uint32_t)__builtin_ctzll(m);
        return p;
    }
    it->mask = 0;
    while (it->bm_cur + 1 < it->bm_end) {
        it->bm_cur++;
        it->block_base += 64;
        m = *it->bm_cur;
        if (m) {
            it->mask = m;
            it->cur_pos = it->block_base + (uint32_t)__builtin_ctzll(m);
            return p;
        }
    }
    return p;
}

FINLINE void classify_block(const __m256i v,
                            const __m256i nib_mask,
                            const __m256i shuf_lo,
                            const __m256i shuf_hi,
                            const __m256i ws_thresh,
                            const __m256i zeroes,
                            uint32_t *struct_out,
                            uint32_t *ws_out,
                            uint32_t *special_out) {
    __m256i lo_nib = _mm256_and_si256(v, nib_mask);
    __m256i hi_nib = _mm256_and_si256(_mm256_srli_epi16(v, 4), nib_mask);
    __m256i cls = _mm256_and_si256(
        _mm256_shuffle_epi8(shuf_lo, lo_nib),
        _mm256_shuffle_epi8(shuf_hi, hi_nib));
    *struct_out = (uint32_t)_mm256_movemask_epi8(
        _mm256_cmpgt_epi8(cls, zeroes));
    *special_out = (uint32_t)_mm256_movemask_epi8(cls);
    *ws_out = (uint32_t)_mm256_movemask_epi8(
        _mm256_cmpgt_epi8(ws_thresh, v));
}

static uint32_t stage1_bitmask(const uint8_t *input, uint32_t length,
                                uint64_t *bm_out, uint32_t bm_cap,
                                uint8_t *dirty_out,
                                uint8_t *any_dirty_out) {
    uint32_t num_blocks = 0;
    uint8_t any_dirty_acc = 0;
    uint64_t prev_in_string = 0;
    uint64_t prev_odd_carry = 0;
    uint64_t prev_scalar = 0;

    const __m256i v_bs    = _mm256_set1_epi8('\\');
    const __m256i v_qt    = _mm256_set1_epi8('"');

    const __m256i nib_mask = _mm256_set1_epi8(0x0F);
    const __m256i zeroes   = _mm256_setzero_si256();
    const __m256i ws_thresh = _mm256_set1_epi8(0x21);

    const __m256i shuf_lo = _mm256_setr_epi8(
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x84, (char)0x82,
        (char)0x81, (char)0x88, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x84, (char)0x82,
        (char)0x81, (char)0x88, (char)0x80, (char)0x80);
    const __m256i shuf_hi = _mm256_setr_epi8(
        (char)0x80, (char)0x80, 0x01, 0x04,
        0, 0x0A, 0, 0x0A,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, 0x01, 0x04,
        0, 0x0A, 0, 0x0A,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80);

    uint32_t offset = 0;
    for (; offset + 64 <= length; offset += 64) {
        __m256i lo = _mm256_loadu_si256((const __m256i *)(input + offset));
        __m256i hi = _mm256_loadu_si256((const __m256i *)(input + offset + 32));

        uint64_t bs_bits =
            (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, v_bs))
            | ((uint64_t)(uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, v_bs)) << 32);

        uint64_t quote_bits =
            (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, v_qt))
            | ((uint64_t)(uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, v_qt)) << 32);

        uint64_t escaped = find_escaped(bs_bits, &prev_odd_carry);

        uint64_t real_quotes = quote_bits & ~escaped;
        uint64_t in_string = prefix_xor(real_quotes) ^ prev_in_string;
        prev_in_string = (uint64_t)((int64_t)in_string >> 63);

        uint32_t s_lo, s_hi, w_lo, w_hi, sp_lo, sp_hi;
        classify_block(lo, nib_mask, shuf_lo, shuf_hi, ws_thresh, zeroes, &s_lo, &w_lo, &sp_lo);
        classify_block(hi, nib_mask, shuf_lo, shuf_hi, ws_thresh, zeroes, &s_hi, &w_hi, &sp_hi);

        uint64_t structurals = ((uint64_t)s_lo | ((uint64_t)s_hi << 32)) & ~in_string;
        uint64_t whitespace = (uint64_t)w_lo | ((uint64_t)w_hi << 32);

        uint64_t is_scalar = ~(in_string | structurals | whitespace | real_quotes);
        uint64_t scalar_starts = is_scalar & ~((is_scalar << 1) | prev_scalar);
        prev_scalar = is_scalar >> 63;

        uint64_t special_bits = (uint64_t)sp_lo | ((uint64_t)sp_hi << 32);
        uint64_t non_ascii =
            (uint32_t)_mm256_movemask_epi8(lo)
            | ((uint64_t)(uint32_t)_mm256_movemask_epi8(hi) << 32);
        uint64_t ctrl_bits = special_bits & ~non_ascii;

        if (UNLIKELY(num_blocks >= bm_cap)) { *any_dirty_out = any_dirty_acc; return num_blocks; }
        dirty_out[num_blocks] = ((bs_bits | ctrl_bits) & in_string) ? 1 : 0;
        any_dirty_acc |= dirty_out[num_blocks];
        bm_out[num_blocks++] = structurals | real_quotes | scalar_starts;
    }

    if (offset < length) {
        uint8_t buf[64] __attribute__((aligned(32)));
        uint32_t remain = length - offset;
        __builtin_memcpy(buf, input + offset, remain);
        __builtin_memset(buf + remain, 0x20, 64 - remain);

        __m256i lo = _mm256_load_si256((const __m256i *)buf);
        __m256i hi = _mm256_load_si256((const __m256i *)(buf + 32));

        uint64_t bs_bits =
            (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, v_bs))
            | ((uint64_t)(uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, v_bs)) << 32);

        uint64_t quote_bits =
            (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, v_qt))
            | ((uint64_t)(uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, v_qt)) << 32);

        uint64_t tail_mask = remain < 64 ? ((1ULL << remain) - 1) : ~0ULL;
        bs_bits &= tail_mask;
        quote_bits &= tail_mask;

        uint64_t escaped = find_escaped(bs_bits, &prev_odd_carry);

        uint64_t real_quotes = quote_bits & ~escaped;
        uint64_t in_string = prefix_xor(real_quotes) ^ prev_in_string;

        uint32_t s_lo, s_hi, w_lo, w_hi, sp_lo, sp_hi;
        classify_block(lo, nib_mask, shuf_lo, shuf_hi, ws_thresh, zeroes, &s_lo, &w_lo, &sp_lo);
        classify_block(hi, nib_mask, shuf_lo, shuf_hi, ws_thresh, zeroes, &s_hi, &w_hi, &sp_hi);

        uint64_t structurals = ((uint64_t)s_lo | ((uint64_t)s_hi << 32)) & ~in_string & tail_mask;
        uint64_t whitespace = ((uint64_t)w_lo | ((uint64_t)w_hi << 32)) & tail_mask;

        uint64_t is_scalar = ~(in_string | structurals | whitespace | real_quotes) & tail_mask;
        uint64_t scalar_starts = is_scalar & ~((is_scalar << 1) | prev_scalar);

        uint64_t special_bits = (uint64_t)sp_lo | ((uint64_t)sp_hi << 32);
        uint64_t non_ascii =
            (uint32_t)_mm256_movemask_epi8(lo)
            | ((uint64_t)(uint32_t)_mm256_movemask_epi8(hi) << 32);
        uint64_t ctrl_bits = special_bits & ~non_ascii;

        if (UNLIKELY(num_blocks >= bm_cap)) { *any_dirty_out = any_dirty_acc; return num_blocks; }
        dirty_out[num_blocks] = ((bs_bits | ctrl_bits) & in_string) ? 1 : 0;
        any_dirty_acc |= dirty_out[num_blocks];
        bm_out[num_blocks++] = structurals | real_quotes | scalar_starts;
    }

    *any_dirty_out = any_dirty_acc;
    return num_blocks;
}

FINLINE uint32_t stage2_core(P *p, SiIter *itp, const uint8_t *dirty,
                             const int all_clean) {
    Frame stack[JBIN_MAX_DEPTH];
    int sp = -1;
    uint32_t value;
    JbinNode *nodes = p->arena->nodes;
    const uint8_t *input = p->input;

    SiIter it = *itp;
    p->last_pos = 0;
    uint32_t nc = p->node_count;
    uint32_t pos;
    uint8_t c;

do_value:
    if (UNLIKELY(!si_has(&it))) {
        p->error = JBIN_ERR_UNEXPECTED;
        p->error_pos = p->last_pos;
        return JBIN_NONE;
    }
    if (UNLIKELY(nc + 3 >= JBIN_MAX_NODES)) {
        p->error = JBIN_ERR_NODES_FULL;
        p->error_pos = p->last_pos;
        return JBIN_NONE;
    }
    pos = si_next(&it);
    c = input[pos];
do_dispatch: ;
    {

        if (c == '"') {
            uint32_t close_pos = si_next(&it);
            uint32_t slen = close_pos - pos - 1;
            if (all_clean || LIKELY(!(dirty[pos >> 6] | dirty[close_pos >> 6]))) {
                value = nc++;
                jbin_node_set(&nodes[value], JBIN_STRING, JBIN_NONE);
                nodes[value].str_off = (pos + 1) | JBIN_INPUT_REF;
                nodes[value].str_len = slen;
            } else {
                p->node_count = nc;
                p->cur = input + pos;
                value = parse_string_fast(p);
                if (UNLIKELY(value == JBIN_NONE)) return JBIN_NONE;
                nc = p->node_count;
            }
            p->last_pos = pos;
            goto have_value;
        }
        if (c == '-' || (uint32_t)(c - '0') <= 9u) {
            if (LIKELY(si_has(&it))) {
                value = parse_number_s2(p, input + pos, nodes, nc);
                if (UNLIKELY(value == JBIN_NONE)) return JBIN_NONE;
                nc++;
            } else {
                p->node_count = nc;
                p->cur = input + pos;
                value = parse_number(p);
                if (UNLIKELY(value == JBIN_NONE)) return JBIN_NONE;
                nc = p->node_count;
                if (UNLIKELY(p->cur < p->end
                    && *p->cur != ',' && *p->cur != '}' && *p->cur != ']'
                    && *p->cur != ' ' && *p->cur != '\n' && *p->cur != '\r' && *p->cur != '\t')) {
                    fail(p, JBIN_ERR_BAD_NUMBER);
                    return JBIN_NONE;
                }
            }
            p->last_pos = pos;
            goto have_value;
        }
        if (c == 't') {
            uint32_t w; __builtin_memcpy(&w, input + pos, 4);
            if (LIKELY(w == 0x65757274u)) {
                value = nc++;
                jbin_node_set(&nodes[value], JBIN_TRUE, JBIN_NONE);
                p->last_pos = pos;
                goto have_value;
            }
            p->cur = input + pos;
            fail(p, JBIN_ERR_BAD_LITERAL);
            return JBIN_NONE;
        }
        if (c == 'f') {
            uint32_t w; __builtin_memcpy(&w, input + pos + 1, 4);
            if (LIKELY(w == 0x65736c61u)) {
                value = nc++;
                jbin_node_set(&nodes[value], JBIN_FALSE, JBIN_NONE);
                p->last_pos = pos;
                goto have_value;
            }
            p->cur = input + pos;
            fail(p, JBIN_ERR_BAD_LITERAL);
            return JBIN_NONE;
        }
        if (c == 'n') {
            uint32_t w; __builtin_memcpy(&w, input + pos, 4);
            if (LIKELY(w == 0x6c6c756eu)) {
                value = nc++;
                jbin_node_set(&nodes[value], JBIN_NULL, JBIN_NONE);
                p->last_pos = pos;
                goto have_value;
            }
            p->cur = input + pos;
            fail(p, JBIN_ERR_BAD_LITERAL);
            return JBIN_NONE;
        }
        if (c == '[') {
            p->cur = input + pos + 1;
            p->last_pos = pos;
            if (UNLIKELY(sp >= JBIN_MAX_DEPTH - 1)) {
                fail(p, JBIN_ERR_DEPTH);
                return JBIN_NONE;
            }
            value = nc++;
            jbin_node_set(&nodes[value], JBIN_ARRAY, JBIN_NONE);
            nodes[value].first_child = JBIN_NONE;
            if (UNLIKELY(!si_has(&it))) {
                p->error = JBIN_ERR_UNEXPECTED;
                p->error_pos = p->last_pos;
                return JBIN_NONE;
            }
            {
                uint32_t fpos = si_pos(&it);
                if (input[fpos] == ']') {
                    p->last_pos = fpos;
                    si_advance(&it);
                    goto have_value;
                }
            }
            sp++;
            stack[sp].node = value;
            stack[sp].link = &nodes[value].first_child;
            stack[sp].is_obj = 0;
            goto do_value;
        }
        if (c == '{') {
            p->cur = input + pos + 1;
            p->last_pos = pos;
            if (UNLIKELY(sp >= JBIN_MAX_DEPTH - 1)) {
                fail(p, JBIN_ERR_DEPTH);
                return JBIN_NONE;
            }
            uint32_t obj = nc++;
            jbin_node_set(&nodes[obj], JBIN_OBJECT, JBIN_NONE);
            nodes[obj].first_child = JBIN_NONE;
            if (UNLIKELY(!si_has(&it))) {
                p->error = JBIN_ERR_UNEXPECTED;
                p->error_pos = p->last_pos;
                return JBIN_NONE;
            }
            uint32_t key;
            {
                uint32_t npos = si_next(&it);
                uint8_t fc = input[npos];
                if (fc == '}') {
                    p->last_pos = npos;
                    value = obj;
                    goto have_value;
                }
                if (UNLIKELY(fc != '"')) {
                    p->error = JBIN_ERR_EXPECTED_KEY;
                    p->error_pos = npos;
                    return JBIN_NONE;
                }
                uint32_t kclose = si_next(&it);
                uint32_t klen = kclose - npos - 1;
                if (all_clean || LIKELY(!(dirty[npos >> 6] | dirty[kclose >> 6]))) {
                    key = nc++;
                    jbin_node_set(&nodes[key], JBIN_STRING, JBIN_NONE);
                    nodes[key].str_off = (npos + 1) | JBIN_INPUT_REF;
                    nodes[key].str_len = klen;
                } else {
                    p->node_count = nc;
                    p->cur = input + npos;
                    key = parse_string_fast(p);
                    if (UNLIKELY(key == JBIN_NONE)) return JBIN_NONE;
                    nc = p->node_count;
                }
            }
            {
                uint32_t cpos = si_next(&it);
                if (UNLIKELY(input[cpos] != ':')) {
                    p->error = JBIN_ERR_EXPECTED_COLON;
                    p->error_pos = cpos;
                    return JBIN_NONE;
                }
            }
            sp++;
            stack[sp].node = obj;
            stack[sp].link = &nodes[obj].first_child;
            stack[sp].is_obj = 1;
            stack[sp].key = key;
            goto do_value;
        }
    }

    p->error = JBIN_ERR_UNEXPECTED;
    p->error_pos = si_pos(&it);
    return JBIN_NONE;

have_value:
    if (sp < 0) {
        if (UNLIKELY(si_has(&it))) {
            p->error = JBIN_ERR_TRAILING;
            p->error_pos = si_pos(&it);
            return JBIN_NONE;
        }
        jbin_node_set(&nodes[value], jbin_node_type(&nodes[value]), JBIN_NONE);
        p->node_count = nc;
        return value;
    }

    if (stack[sp].is_obj) {
        uint32_t key = stack[sp].key;
        jbin_node_set(&nodes[key], jbin_node_type(&nodes[key]), value);
        LINK_SET(stack[sp].link, key);
        stack[sp].link = &nodes[value].type_next;

        if (UNLIKELY(!si_has(&it))) {
            p->error = JBIN_ERR_UNEXPECTED;
            p->error_pos = p->last_pos;
            return JBIN_NONE;
        }
        {
            uint32_t npos = si_next(&it);
            if (input[npos] == '}') {
                p->last_pos = npos;
                LINK_SET(stack[sp].link, JBIN_NONE);
                value = stack[sp].node;
                sp--;
                goto have_value;
            }
            if (UNLIKELY(input[npos] != ',')) {
                p->error = JBIN_ERR_UNEXPECTED;
                p->error_pos = npos;
                return JBIN_NONE;
            }
        }

        {
            uint32_t npos = si_next(&it);
            if (UNLIKELY(input[npos] != '"')) {
                p->error = JBIN_ERR_EXPECTED_KEY;
                p->error_pos = npos;
                return JBIN_NONE;
            }
            uint32_t kclose = si_next(&it);
            uint32_t klen = kclose - npos - 1;
            if (all_clean || LIKELY(!(dirty[npos >> 6] | dirty[kclose >> 6]))) {
                key = nc++;
                jbin_node_set(&nodes[key], JBIN_STRING, JBIN_NONE);
                nodes[key].str_off = (npos + 1) | JBIN_INPUT_REF;
                nodes[key].str_len = klen;
            } else {
                p->node_count = nc;
                p->cur = input + npos;
                key = parse_string_fast(p);
                if (UNLIKELY(key == JBIN_NONE)) return JBIN_NONE;
                nc = p->node_count;
            }
        }

        {
            uint32_t cpos = si_next(&it);
            if (UNLIKELY(input[cpos] != ':')) {
                p->error = JBIN_ERR_EXPECTED_COLON;
                p->error_pos = cpos;
                return JBIN_NONE;
            }
        }

        stack[sp].key = key;
        /* Fused: fetch next value structural without going through do_value */
        if (UNLIKELY(!si_has(&it))) {
            p->error = JBIN_ERR_UNEXPECTED;
            p->error_pos = p->last_pos;
            return JBIN_NONE;
        }
        if (UNLIKELY(nc + 3 >= JBIN_MAX_NODES)) {
            p->error = JBIN_ERR_NODES_FULL;
            p->error_pos = p->last_pos;
            return JBIN_NONE;
        }
        pos = si_next(&it);
        c = input[pos];
        goto do_dispatch;
    }

    LINK_SET(stack[sp].link, value);
    stack[sp].link = &nodes[value].type_next;

    if (UNLIKELY(!si_has(&it))) {
        p->error = JBIN_ERR_UNEXPECTED;
        p->error_pos = p->last_pos;
        return JBIN_NONE;
    }
    {
        uint32_t npos = si_next(&it);
        if (input[npos] == ']') {
            p->last_pos = npos;
            LINK_SET(stack[sp].link, JBIN_NONE);
            value = stack[sp].node;
            sp--;
            goto have_value;
        }
        if (UNLIKELY(input[npos] != ',')) {
            p->error = JBIN_ERR_UNEXPECTED;
            p->error_pos = npos;
            return JBIN_NONE;
        }
    }
    /* Fused: fetch next value structural without going through do_value */
    if (UNLIKELY(!si_has(&it))) {
        p->error = JBIN_ERR_UNEXPECTED;
        p->error_pos = p->last_pos;
        return JBIN_NONE;
    }
    if (UNLIKELY(nc + 3 >= JBIN_MAX_NODES)) {
        p->error = JBIN_ERR_NODES_FULL;
        p->error_pos = p->last_pos;
        return JBIN_NONE;
    }
    pos = si_next(&it);
    c = input[pos];
    goto do_dispatch;
}

static uint32_t stage2_bitmask(P *p, SiIter *itp, const uint8_t *dirty) {
    return stage2_core(p, itp, dirty, 0);
}

static uint32_t stage2_clean(P *p, SiIter *itp) {
    return stage2_core(p, itp, NULL, 1);
}

/* --- Tape-based stage2 (sequential writes, no tree construction) --- */

typedef struct {
    uint32_t open_idx;
    uint8_t  is_obj;
} TapeFrame;

FINLINE int tape_str_dirty(const uint8_t *s, uint32_t len) {
    const uint64_t LO = 0x0101010101010101ULL;
    const uint64_t HI = 0x8080808080808080ULL;
    const uint64_t BS = 0x5C5C5C5C5C5C5C5CULL;
    const uint64_t SP = 0x2020202020202020ULL;
    uint32_t i = 0;
    for (; i + 8 <= len; i += 8) {
        uint64_t v;
        __builtin_memcpy(&v, s + i, 8);
        /* backslash: haszero(v ^ 0x5C) */
        uint64_t xbs = v ^ BS;
        uint64_t has_bs = (xbs - LO) & ~xbs & HI;
        /* control chars < 0x20, no false positives for >= 0x80 */
        uint64_t has_ctrl = ~((v | HI) - SP) & ~v & HI;
        if (has_bs | has_ctrl) return 1;
    }
    for (; i < len; i++) {
        uint8_t c = s[i];
        if (c == '\\' || c < 0x20)
            return 1;
    }
    return 0;
}

/*
 * Inline string handler for strings < 32 bytes in dirty blocks.
 * Uses a single AVX2 load to detect backslashes, then:
 *   - No backslash → zero-copy reference
 *   - Simple escapes → copy+decode inline using bitmask-guided segments
 *   - Unicode escapes → return 0 (fall back to tape_string_fused)
 * Returns: 1 = handled, 0 = needs fused, -1 = error.
 */
FINLINE int tape_string_inline(P *p, const uint8_t *input,
                                 uint64_t *tape, uint32_t *tc_p,
                                 uint32_t pos, uint32_t slen,
                                 __m256i vbs, __m256i vxor, __m256i vthresh) {
    if (slen >= 32 || pos + 33 > (uint32_t)(p->end - p->input))
        return 0;
    const uint8_t *s = input + pos + 1;
    __m256i v = _mm256_loadu_si256((const __m256i *)s);
    uint32_t slen_mask = slen ? (1u << slen) - 1 : 0;
    uint32_t bs = (uint32_t)_mm256_movemask_epi8(
        _mm256_cmpeq_epi8(v, vbs)) & slen_mask;
    uint32_t cm = (uint32_t)_mm256_movemask_epi8(
        _mm256_cmpgt_epi8(vthresh, _mm256_xor_si256(v, vxor))) & slen_mask;
    if (UNLIKELY(cm)) {
        p->cur = s + (uint32_t)__builtin_ctz(cm);
        fail(p, JBIN_ERR_CONTROL_CHAR);
        return -1;
    }
    if (LIKELY(!bs)) {
        /* No backslash → zero-copy */
        uint32_t na = (uint32_t)_mm256_movemask_epi8(v) & slen_mask;
        if (UNLIKELY(na) && !validate_utf8(s, slen)) {
            p->cur = s;
            fail(p, JBIN_ERR_BAD_UTF8);
            return -1;
        }
        tape[(*tc_p)++] = JBIN_TAPE_MAKE_STR(1, pos + 1, slen);
        return 1;
    }
    /* Has backslash → copy segments between escapes using bitmask */
    uint32_t arena_start = p->arena->string_used;
    if (UNLIKELY(arena_start + slen >= JBIN_MAX_STRING)) return 0;
    uint8_t *dst_base = (uint8_t *)p->arena->strings + arena_start;
    uint8_t *dst = dst_base;
    uint32_t scan = 0;
    uint32_t esc_bits = bs;
    while (esc_bits) {
        uint32_t bp = (uint32_t)__builtin_ctz(esc_bits);
        if (bp > scan) { __builtin_memcpy(dst, s + scan, bp - scan); dst += bp - scan; }
        uint8_t rep = ESC[s[bp + 1]];
        if (UNLIKELY(!rep)) return 0; /* unicode or bad escape → fall back */
        *dst++ = rep;
        scan = bp + 2;
        esc_bits = _blsr_u64(esc_bits);
    }
    if (scan < slen) { __builtin_memcpy(dst, s + scan, slen - scan); dst += slen - scan; }
    uint32_t out_slen = (uint32_t)(dst - dst_base);
    p->arena->string_used = arena_start + out_slen;
    uint32_t na = (uint32_t)_mm256_movemask_epi8(v) & slen_mask;
    if (UNLIKELY(na) && !validate_utf8(dst_base, out_slen)) {
        p->arena->string_used = arena_start;
        p->cur = s;
        fail(p, JBIN_ERR_BAD_UTF8);
        return -1;
    }
    tape[(*tc_p)++] = JBIN_TAPE_MAKE_STR(0, arena_start, out_slen);
    return 1;
}

static int tape_string_fused(P *p, uint64_t *tape, uint32_t *tc_p,
                              uint32_t known_slen) {
    if (UNLIKELY(p->cur >= p->end || *p->cur != '"')) {
        fail(p, JBIN_ERR_UNEXPECTED);
        return 0;
    }

    /* Fast path: short string with no backslash/control → zero-copy */
    if (known_slen <= 16) {
        const uint8_t *s = p->cur + 1;
        if (!tape_str_dirty(s, known_slen)) {
            uint32_t off = (uint32_t)(p->cur - p->input) + 1;
            tape[(*tc_p)++] = JBIN_TAPE_MAKE_STR(1, off, known_slen);
            p->cur = s + known_slen + 1;
            return 1;
        }
    }

    p->cur++;

    uint32_t arena_start = p->arena->string_used;
    uint8_t *dst_base = (uint8_t *)p->arena->strings + arena_start;
    uint8_t *dst = dst_base;
    uint32_t capacity = JBIN_MAX_STRING - arena_start;
    const uint8_t *src = p->cur;
    const uint8_t *end = p->end;
    uint8_t *dst_limit = dst_base + capacity - 31;
    int has_non_ascii = 0;

    const __m256i vquote = _mm256_set1_epi8('"');
    const __m256i vbackslash = _mm256_set1_epi8('\\');
    const __m256i vxor = _mm256_set1_epi8((char)0x80);
    const __m256i vthresh = _mm256_set1_epi8((char)0xA0);

    while (src + 32 <= end && dst <= dst_limit) {
        __m256i v = _mm256_loadu_si256((const __m256i *)src);
        uint32_t qmask = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vquote));
        uint32_t bmask = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vbackslash));

        if (LIKELY(!(qmask | bmask))) {
            _mm256_storeu_si256((__m256i *)dst, v);
            has_non_ascii |= (int)_mm256_movemask_epi8(v);
            src += 32;
            dst += 32;
            continue;
        }

        uint32_t cmask = (uint32_t)_mm256_movemask_epi8(
            _mm256_cmpgt_epi8(vthresh, _mm256_xor_si256(v, vxor)));
        uint32_t fpos = (uint32_t)__builtin_ctz(qmask | bmask | cmask);

        has_non_ascii |= (int)_mm256_movemask_epi8(v);
        _mm256_storeu_si256((__m256i *)dst, v);
        dst += fpos;
        src += fpos;

        if (UNLIKELY(cmask & (1u << fpos))) {
            p->cur = src;
            fail(p, JBIN_ERR_CONTROL_CHAR);
            return 0;
        }

        if (qmask & (1u << fpos)) {
            uint32_t slen = (uint32_t)(dst - dst_base);
            p->arena->string_used = arena_start + slen;
            if (UNLIKELY(has_non_ascii && !validate_utf8(dst_base, slen))) {
                p->cur = src;
                fail(p, JBIN_ERR_BAD_UTF8);
                return 0;
            }
            tape[(*tc_p)++] = JBIN_TAPE_MAKE_STR(0, arena_start, slen);
            p->cur = src + 1;
            return 1;
        }

        if (UNLIKELY(src + 1 >= end)) {
            p->cur = src;
            fail(p, JBIN_ERR_UNTERMINATED_STRING);
            return 0;
        }
        uint8_t esc_ch = src[1];
        src += 2;
        uint8_t rep = ESC[esc_ch];
        if (LIKELY(rep)) {
            *dst++ = rep;
            while (src + 1 < end && *src == '\\') {
                uint8_t nr = ESC[src[1]];
                if (!nr) break;
                *dst++ = nr;
                src += 2;
            }
            continue;
        }
        if (LIKELY(esc_ch == 'u')) {
            if (UNLIKELY(src + 4 > end)) {
                p->cur = src - 2;
                fail(p, JBIN_ERR_BAD_UNICODE);
                return 0;
            }
            uint8_t a = HEX[src[0]], b = HEX[src[1]];
            uint8_t hc = HEX[src[2]], d = HEX[src[3]];
            if (UNLIKELY((a | b | hc | d) & 0x80)) {
                p->cur = src - 2;
                fail(p, JBIN_ERR_BAD_UNICODE);
                return 0;
            }
            uint32_t cp = ((uint32_t)a << 12) | ((uint32_t)b << 8)
                        | ((uint32_t)hc << 4) | d;
            src += 4;
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                if (UNLIKELY(src + 6 > end || src[0] != '\\' || src[1] != 'u')) {
                    p->cur = src - 6;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return 0;
                }
                src += 2;
                uint8_t la = HEX[src[0]], lb = HEX[src[1]];
                uint8_t lc = HEX[src[2]], ld = HEX[src[3]];
                if (UNLIKELY((la | lb | lc | ld) & 0x80)) {
                    p->cur = src - 2;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return 0;
                }
                uint32_t low = ((uint32_t)la << 12) | ((uint32_t)lb << 8)
                             | ((uint32_t)lc << 4) | ld;
                src += 4;
                if (UNLIKELY(low < 0xDC00 || low > 0xDFFF)) {
                    p->cur = src - 6;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return 0;
                }
                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
            } else if (UNLIKELY(cp >= 0xDC00 && cp <= 0xDFFF)) {
                p->cur = src - 6;
                fail(p, JBIN_ERR_BAD_UNICODE);
                return 0;
            }
            if (UNLIKELY((uint32_t)(dst - dst_base) + 4 > capacity)) {
                p->cur = src;
                fail(p, JBIN_ERR_STRING_FULL);
                return 0;
            }
            if (cp <= 0x7F) {
                *dst++ = (uint8_t)cp;
            } else if (cp <= 0x7FF) {
                *dst++ = (uint8_t)(0xC0 | (cp >> 6));
                *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
            } else if (cp <= 0xFFFF) {
                *dst++ = (uint8_t)(0xE0 | (cp >> 12));
                *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
            } else {
                *dst++ = (uint8_t)(0xF0 | (cp >> 18));
                *dst++ = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
                *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
            }
        } else {
            p->cur = src - 2;
            fail(p, JBIN_ERR_BAD_ESCAPE);
            return 0;
        }
    }

    while (src < end) {
        if (UNLIKELY((uint32_t)(dst - dst_base) >= capacity)) {
            p->cur = src;
            fail(p, JBIN_ERR_STRING_FULL);
            return 0;
        }
        uint8_t ch = *src;
        if (ch == '"') {
            uint32_t slen = (uint32_t)(dst - dst_base);
            p->arena->string_used = arena_start + slen;
            if (UNLIKELY(has_non_ascii && !validate_utf8(dst_base, slen))) {
                p->cur = src;
                fail(p, JBIN_ERR_BAD_UTF8);
                return 0;
            }
            tape[(*tc_p)++] = JBIN_TAPE_MAKE_STR(0, arena_start, slen);
            p->cur = src + 1;
            return 1;
        }
        if (UNLIKELY(ch < 0x20)) {
            p->cur = src;
            fail(p, JBIN_ERR_CONTROL_CHAR);
            return 0;
        }
        if (ch == '\\') {
            if (UNLIKELY(src + 1 >= end)) {
                p->cur = src;
                fail(p, JBIN_ERR_UNTERMINATED_STRING);
                return 0;
            }
            uint8_t esc_ch = src[1];
            src += 2;
            uint8_t rep = ESC[esc_ch];
            if (LIKELY(rep)) {
                *dst++ = rep;
            } else if (LIKELY(esc_ch == 'u')) {
                if (UNLIKELY(src + 4 > end)) {
                    p->cur = src - 2;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return 0;
                }
                uint8_t a = HEX[src[0]], b = HEX[src[1]];
                uint8_t hc = HEX[src[2]], d = HEX[src[3]];
                if (UNLIKELY((a | b | hc | d) & 0x80)) {
                    p->cur = src - 2;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return 0;
                }
                uint32_t cp = ((uint32_t)a << 12) | ((uint32_t)b << 8)
                            | ((uint32_t)hc << 4) | d;
                src += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (UNLIKELY(src + 6 > end || src[0] != '\\' || src[1] != 'u')) {
                        p->cur = src - 6;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return 0;
                    }
                    src += 2;
                    uint8_t la = HEX[src[0]], lb = HEX[src[1]];
                    uint8_t lc = HEX[src[2]], ld = HEX[src[3]];
                    if (UNLIKELY((la | lb | lc | ld) & 0x80)) {
                        p->cur = src - 2;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return 0;
                    }
                    uint32_t low = ((uint32_t)la << 12) | ((uint32_t)lb << 8)
                                 | ((uint32_t)lc << 4) | ld;
                    src += 4;
                    if (UNLIKELY(low < 0xDC00 || low > 0xDFFF)) {
                        p->cur = src - 6;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return 0;
                    }
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                } else if (UNLIKELY(cp >= 0xDC00 && cp <= 0xDFFF)) {
                    p->cur = src - 6;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return 0;
                }
                if (UNLIKELY((uint32_t)(dst - dst_base) + 4 > capacity)) {
                    p->cur = src;
                    fail(p, JBIN_ERR_STRING_FULL);
                    return 0;
                }
                if (cp <= 0x7F) {
                    *dst++ = (uint8_t)cp;
                } else if (cp <= 0x7FF) {
                    *dst++ = (uint8_t)(0xC0 | (cp >> 6));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                } else if (cp <= 0xFFFF) {
                    *dst++ = (uint8_t)(0xE0 | (cp >> 12));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                } else {
                    *dst++ = (uint8_t)(0xF0 | (cp >> 18));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                }
            } else {
                p->cur = src - 2;
                fail(p, JBIN_ERR_BAD_ESCAPE);
                return 0;
            }
            continue;
        }
        has_non_ascii |= (ch >> 7);
        *dst++ = ch;
        src++;
    }

    p->cur = src;
    fail(p, JBIN_ERR_UNTERMINATED_STRING);
    return 0;
}

#if 0 /* deferred batch: experimentally slower for dirty-heavy files */
NOINLINE int tape_string_batch(P *p, uint64_t *tape) {
    const __m256i vquote = _mm256_set1_epi8('"');
    const __m256i vbackslash = _mm256_set1_epi8('\\');
    const __m256i vxor = _mm256_set1_epi8((char)0x80);
    const __m256i vthresh = _mm256_set1_epi8((char)0xA0);

    for (uint32_t di = 0; di < p->dirty_count; di++) {
        /* Prefetch next dirty string's input data */
        if (di + 4 < p->dirty_count) {
            uint32_t noff = JBIN_TAPE_OFFSET(tape[p->dirty_buf[di + 4]]);
            __builtin_prefetch(p->input + noff, 0, 1);
        }

        uint32_t tape_idx = p->dirty_buf[di];
        uint64_t entry = tape[tape_idx];
        uint32_t str_off = JBIN_TAPE_OFFSET(entry);

        const uint8_t *src = p->input + str_off;
        const uint8_t *end = p->end;
        uint32_t arena_start = p->arena->string_used;
        uint8_t *dst_base = (uint8_t *)p->arena->strings + arena_start;
        uint8_t *dst = dst_base;
        uint32_t capacity = JBIN_MAX_STRING - arena_start;
        uint8_t *dst_limit = dst_base + capacity - 31;
        int has_non_ascii = 0;

        /* AVX2 string processing loop (constants already loaded above) */
        while (src + 32 <= end && dst <= dst_limit) {
            __m256i v = _mm256_loadu_si256((const __m256i *)src);
            uint32_t qmask = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vquote));
            uint32_t bmask = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vbackslash));

            if (LIKELY(!(qmask | bmask))) {
                _mm256_storeu_si256((__m256i *)dst, v);
                has_non_ascii |= (int)_mm256_movemask_epi8(v);
                src += 32;
                dst += 32;
                continue;
            }

            uint32_t cmask = (uint32_t)_mm256_movemask_epi8(
                _mm256_cmpgt_epi8(vthresh, _mm256_xor_si256(v, vxor)));
            uint32_t fpos = (uint32_t)__builtin_ctz(qmask | bmask | cmask);

            has_non_ascii |= (int)_mm256_movemask_epi8(v);
            _mm256_storeu_si256((__m256i *)dst, v);
            dst += fpos;
            src += fpos;

            if (UNLIKELY(cmask & (1u << fpos))) {
                p->cur = src;
                fail(p, JBIN_ERR_CONTROL_CHAR);
                return 0;
            }

            if (qmask & (1u << fpos)) {
                /* Found closing quote */
                uint32_t slen = (uint32_t)(dst - dst_base);
                p->arena->string_used = arena_start + slen;
                if (UNLIKELY(has_non_ascii && !validate_utf8(dst_base, slen))) {
                    p->cur = src;
                    fail(p, JBIN_ERR_BAD_UTF8);
                    return 0;
                }
                tape[tape_idx] = JBIN_TAPE_MAKE_STR(0, arena_start, slen);
                goto next_dirty;
            }

            /* Backslash escape */
            if (UNLIKELY(src + 1 >= end)) {
                p->cur = src;
                fail(p, JBIN_ERR_UNTERMINATED_STRING);
                return 0;
            }
            uint8_t esc_ch = src[1];
            src += 2;
            uint8_t rep = ESC[esc_ch];
            if (LIKELY(rep)) {
                *dst++ = rep;
                while (src + 1 < end && *src == '\\') {
                    uint8_t nr = ESC[src[1]];
                    if (!nr) break;
                    *dst++ = nr;
                    src += 2;
                }
                continue;
            }
            if (LIKELY(esc_ch == 'u')) {
                if (UNLIKELY(src + 4 > end)) {
                    p->cur = src - 2;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return 0;
                }
                uint8_t a = HEX[src[0]], b = HEX[src[1]];
                uint8_t hc = HEX[src[2]], d = HEX[src[3]];
                if (UNLIKELY((a | b | hc | d) & 0x80)) {
                    p->cur = src - 2;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return 0;
                }
                uint32_t cp = ((uint32_t)a << 12) | ((uint32_t)b << 8)
                            | ((uint32_t)hc << 4) | d;
                src += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (UNLIKELY(src + 6 > end || src[0] != '\\' || src[1] != 'u')) {
                        p->cur = src - 6;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return 0;
                    }
                    src += 2;
                    uint8_t la = HEX[src[0]], lb = HEX[src[1]];
                    uint8_t lc = HEX[src[2]], ld = HEX[src[3]];
                    if (UNLIKELY((la | lb | lc | ld) & 0x80)) {
                        p->cur = src - 2;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return 0;
                    }
                    uint32_t low = ((uint32_t)la << 12) | ((uint32_t)lb << 8)
                                 | ((uint32_t)lc << 4) | ld;
                    src += 4;
                    if (UNLIKELY(low < 0xDC00 || low > 0xDFFF)) {
                        p->cur = src - 6;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return 0;
                    }
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                } else if (UNLIKELY(cp >= 0xDC00 && cp <= 0xDFFF)) {
                    p->cur = src - 6;
                    fail(p, JBIN_ERR_BAD_UNICODE);
                    return 0;
                }
                if (UNLIKELY((uint32_t)(dst - dst_base) + 4 > capacity)) {
                    p->cur = src;
                    fail(p, JBIN_ERR_STRING_FULL);
                    return 0;
                }
                if (cp <= 0x7F) {
                    *dst++ = (uint8_t)cp;
                } else if (cp <= 0x7FF) {
                    *dst++ = (uint8_t)(0xC0 | (cp >> 6));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                } else if (cp <= 0xFFFF) {
                    *dst++ = (uint8_t)(0xE0 | (cp >> 12));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                } else {
                    *dst++ = (uint8_t)(0xF0 | (cp >> 18));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                    *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                }
            } else {
                p->cur = src - 2;
                fail(p, JBIN_ERR_BAD_ESCAPE);
                return 0;
            }
        }

        /* Scalar tail */
        while (src < end) {
            if (UNLIKELY((uint32_t)(dst - dst_base) >= capacity)) {
                p->cur = src;
                fail(p, JBIN_ERR_STRING_FULL);
                return 0;
            }
            uint8_t ch = *src;
            if (ch == '"') {
                uint32_t slen = (uint32_t)(dst - dst_base);
                p->arena->string_used = arena_start + slen;
                if (UNLIKELY(has_non_ascii && !validate_utf8(dst_base, slen))) {
                    p->cur = src;
                    fail(p, JBIN_ERR_BAD_UTF8);
                    return 0;
                }
                tape[tape_idx] = JBIN_TAPE_MAKE_STR(0, arena_start, slen);
                goto next_dirty;
            }
            if (UNLIKELY(ch < 0x20)) {
                p->cur = src;
                fail(p, JBIN_ERR_CONTROL_CHAR);
                return 0;
            }
            if (ch == '\\') {
                if (UNLIKELY(src + 1 >= end)) {
                    p->cur = src;
                    fail(p, JBIN_ERR_UNTERMINATED_STRING);
                    return 0;
                }
                uint8_t esc_ch = src[1];
                src += 2;
                uint8_t rep = ESC[esc_ch];
                if (LIKELY(rep)) {
                    *dst++ = rep;
                } else if (LIKELY(esc_ch == 'u')) {
                    if (UNLIKELY(src + 4 > end)) {
                        p->cur = src - 2;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return 0;
                    }
                    uint8_t a = HEX[src[0]], b = HEX[src[1]];
                    uint8_t hc = HEX[src[2]], d = HEX[src[3]];
                    if (UNLIKELY((a | b | hc | d) & 0x80)) {
                        p->cur = src - 2;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return 0;
                    }
                    uint32_t cp = ((uint32_t)a << 12) | ((uint32_t)b << 8)
                                | ((uint32_t)hc << 4) | d;
                    src += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (UNLIKELY(src + 6 > end || src[0] != '\\' || src[1] != 'u')) {
                            p->cur = src - 6;
                            fail(p, JBIN_ERR_BAD_UNICODE);
                            return 0;
                        }
                        src += 2;
                        uint8_t la = HEX[src[0]], lb = HEX[src[1]];
                        uint8_t lc = HEX[src[2]], ld = HEX[src[3]];
                        if (UNLIKELY((la | lb | lc | ld) & 0x80)) {
                            p->cur = src - 2;
                            fail(p, JBIN_ERR_BAD_UNICODE);
                            return 0;
                        }
                        uint32_t low = ((uint32_t)la << 12) | ((uint32_t)lb << 8)
                                     | ((uint32_t)lc << 4) | ld;
                        src += 4;
                        if (UNLIKELY(low < 0xDC00 || low > 0xDFFF)) {
                            p->cur = src - 6;
                            fail(p, JBIN_ERR_BAD_UNICODE);
                            return 0;
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else if (UNLIKELY(cp >= 0xDC00 && cp <= 0xDFFF)) {
                        p->cur = src - 6;
                        fail(p, JBIN_ERR_BAD_UNICODE);
                        return 0;
                    }
                    if (UNLIKELY((uint32_t)(dst - dst_base) + 4 > capacity)) {
                        p->cur = src;
                        fail(p, JBIN_ERR_STRING_FULL);
                        return 0;
                    }
                    if (cp <= 0x7F) {
                        *dst++ = (uint8_t)cp;
                    } else if (cp <= 0x7FF) {
                        *dst++ = (uint8_t)(0xC0 | (cp >> 6));
                        *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                    } else if (cp <= 0xFFFF) {
                        *dst++ = (uint8_t)(0xE0 | (cp >> 12));
                        *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                        *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                    } else {
                        *dst++ = (uint8_t)(0xF0 | (cp >> 18));
                        *dst++ = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
                        *dst++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                        *dst++ = (uint8_t)(0x80 | (cp & 0x3F));
                    }
                } else {
                    p->cur = src - 2;
                    fail(p, JBIN_ERR_BAD_ESCAPE);
                    return 0;
                }
                continue;
            }
            has_non_ascii |= (ch >> 7);
            *dst++ = ch;
            src++;
        }

        p->cur = src;
        fail(p, JBIN_ERR_UNTERMINATED_STRING);
        return 0;

    next_dirty: ;
    }
    return 1;
}
#endif /* deferred batch */

FINLINE int parse_number_s2_tape(P *p, const uint8_t *start,
                                  uint64_t *tape, uint32_t *tc_p) {
    const uint8_t *cur = start;

    if (*cur == '-') cur++;

    if (UNLIKELY((uint32_t)(*cur - '0') > 9u)) {
        p->cur = cur;
        fail(p, JBIN_ERR_BAD_NUMBER);
        return 0;
    }

    if (*cur == '0') {
        cur++;
        if (UNLIKELY((uint32_t)(*cur - '0') <= 9u)) {
            p->cur = cur;
            fail(p, JBIN_ERR_BAD_NUMBER);
            return 0;
        }
    } else {
        cur++;
        skip_digits_nobound(&cur);
    }

    if (*cur == '.') {
        cur++;
        if (UNLIKELY((uint32_t)(*cur - '0') > 9u)) {
            p->cur = cur;
            fail(p, JBIN_ERR_BAD_NUMBER);
            return 0;
        }
        cur++;
        skip_digits_nobound(&cur);
    }

    if (*cur == 'e' || *cur == 'E') {
        cur++;
        if (*cur == '+' || *cur == '-') cur++;
        if (UNLIKELY((uint32_t)(*cur - '0') > 9u)) {
            p->cur = cur;
            fail(p, JBIN_ERR_BAD_NUMBER);
            return 0;
        }
        do { cur++; } while ((uint32_t)(*cur - '0') <= 9u);
    }

    if (UNLIKELY(*cur > 0x20 && *cur != ',' && *cur != ']' && *cur != '}')) {
        p->cur = cur;
        fail(p, JBIN_ERR_BAD_NUMBER);
        return 0;
    }

    uint32_t off = (uint32_t)(start - p->input);
    uint32_t len = (uint32_t)(cur - start);
    tape[(*tc_p)++] = JBIN_TAPE_MAKE_NUM(1, off, len);
    return 1;
}

FINLINE int parse_number_tape_bounded(P *p, uint64_t *tape, uint32_t *tc_p) {
    const uint8_t *start = p->cur;

    if (p->cur < p->end && *p->cur == '-') p->cur++;

    if (UNLIKELY(p->cur >= p->end || (uint32_t)(*p->cur - '0') > 9u)) {
        fail(p, JBIN_ERR_BAD_NUMBER);
        return 0;
    }

    if (*p->cur == '0') {
        p->cur++;
        if (UNLIKELY(p->cur < p->end && (uint32_t)(*p->cur - '0') <= 9u)) {
            fail(p, JBIN_ERR_BAD_NUMBER);
            return 0;
        }
    } else {
        p->cur++;
        skip_digits(&p->cur, p->end);
    }

    if (p->cur < p->end && *p->cur == '.') {
        p->cur++;
        if (UNLIKELY(p->cur >= p->end || (uint32_t)(*p->cur - '0') > 9u)) {
            fail(p, JBIN_ERR_BAD_NUMBER);
            return 0;
        }
        p->cur++;
        skip_digits(&p->cur, p->end);
    }

    if (p->cur < p->end && (*p->cur == 'e' || *p->cur == 'E')) {
        p->cur++;
        if (p->cur < p->end && (*p->cur == '+' || *p->cur == '-'))
            p->cur++;
        if (UNLIKELY(p->cur >= p->end || (uint32_t)(*p->cur - '0') > 9u)) {
            fail(p, JBIN_ERR_BAD_NUMBER);
            return 0;
        }
        do { p->cur++; } while (p->cur < p->end && (uint32_t)(*p->cur - '0') <= 9u);
    }

    if (UNLIKELY(p->cur < p->end
        && *p->cur != ',' && *p->cur != '}' && *p->cur != ']'
        && *p->cur != ' ' && *p->cur != '\n' && *p->cur != '\r' && *p->cur != '\t')) {
        fail(p, JBIN_ERR_BAD_NUMBER);
        return 0;
    }

    uint32_t off = (uint32_t)(start - p->input);
    uint32_t len = (uint32_t)(p->cur - start);
    tape[(*tc_p)++] = JBIN_TAPE_MAKE_NUM(1, off, len);
    return 1;
}

FINLINE uint32_t stage2_tape_core(P *p, SiIter *itp, const uint8_t *dirty,
                                   const int all_clean) {
    TapeFrame stack[JBIN_MAX_DEPTH];
    int sp = -1;
    uint64_t *tape = p->arena->tape;
    const uint8_t *input = p->input;

    /* AVX2 constants for inline string fast-check */
    const __m256i vbs_s2 = _mm256_set1_epi8('\\');
    const __m256i vxor_s2 = _mm256_set1_epi8((char)0x80);
    const __m256i vthresh_s2 = _mm256_set1_epi8((char)0xA0);

    SiIter it = *itp;
    uint32_t last_pos = 0;
    uint32_t tc = 0;
    uint32_t pos;
    uint8_t c;

do_value:
    if (UNLIKELY(!si_has(&it))) {
        p->error = JBIN_ERR_UNEXPECTED;
        p->error_pos = last_pos;
        return JBIN_NONE;
    }
    if (UNLIKELY(tc + 3 >= JBIN_MAX_TAPE)) {
        p->error = JBIN_ERR_NODES_FULL;
        p->error_pos = last_pos;
        return JBIN_NONE;
    }
    pos = si_next(&it);
    c = input[pos];
do_dispatch: ;
    {
        if (c == '"') {
            uint32_t close_pos = si_next(&it);
            uint32_t slen = close_pos - pos - 1;
            if (all_clean || LIKELY(!(dirty[pos >> 6] | dirty[close_pos >> 6]))) {
                tape[tc++] = JBIN_TAPE_MAKE_STR(1, pos + 1, slen);
            } else {
                int fast = tape_string_inline(p, input, tape, &tc, pos, slen,
                                                  vbs_s2, vxor_s2, vthresh_s2);
                if (fast == 0) {
                    p->cur = input + pos;
                    if (UNLIKELY(!tape_string_fused(p, tape, &tc, slen)))
                        return JBIN_NONE;
                } else if (UNLIKELY(fast < 0))
                    return JBIN_NONE;
            }
            last_pos = pos;
            goto have_value;
        }
        if (c == '-' || (uint32_t)(c - '0') <= 9u) {
            if (LIKELY(si_has(&it))) {
                if (UNLIKELY(!parse_number_s2_tape(p, input + pos, tape, &tc)))
                    return JBIN_NONE;
            } else {
                p->cur = input + pos;
                if (UNLIKELY(!parse_number_tape_bounded(p, tape, &tc)))
                    return JBIN_NONE;
            }
            last_pos = pos;
            goto have_value;
        }
        if (c == 't') {
            uint32_t w; __builtin_memcpy(&w, input + pos, 4);
            if (LIKELY(w == 0x65757274u)) {
                tape[tc++] = JBIN_TAPE_MAKE_LIT(JBIN_TAPE_TRUE);
                last_pos = pos;
                goto have_value;
            }
            p->cur = input + pos;
            fail(p, JBIN_ERR_BAD_LITERAL);
            return JBIN_NONE;
        }
        if (c == 'f') {
            uint32_t w; __builtin_memcpy(&w, input + pos + 1, 4);
            if (LIKELY(w == 0x65736c61u)) {
                tape[tc++] = JBIN_TAPE_MAKE_LIT(JBIN_TAPE_FALSE);
                last_pos = pos;
                goto have_value;
            }
            p->cur = input + pos;
            fail(p, JBIN_ERR_BAD_LITERAL);
            return JBIN_NONE;
        }
        if (c == 'n') {
            uint32_t w; __builtin_memcpy(&w, input + pos, 4);
            if (LIKELY(w == 0x6c6c756eu)) {
                tape[tc++] = JBIN_TAPE_MAKE_LIT(JBIN_TAPE_NULL);
                last_pos = pos;
                goto have_value;
            }
            p->cur = input + pos;
            fail(p, JBIN_ERR_BAD_LITERAL);
            return JBIN_NONE;
        }
        if (c == '[') {
            last_pos = pos;
            if (UNLIKELY(sp >= JBIN_MAX_DEPTH - 1)) {
                p->cur = input + pos;
                fail(p, JBIN_ERR_DEPTH);
                return JBIN_NONE;
            }
            uint32_t open_idx = tc;
            tape[tc++] = JBIN_TAPE_MAKE_OPEN(JBIN_TAPE_AOPEN);
            if (UNLIKELY(!si_has(&it))) {
                p->error = JBIN_ERR_UNEXPECTED;
                p->error_pos = pos;
                return JBIN_NONE;
            }
            {
                uint32_t fpos = si_pos(&it);
                uint8_t fc = input[fpos];
                if (fc == ']') {
                    last_pos = fpos;
                    si_advance(&it);
                    tape[open_idx] = ((uint64_t)JBIN_TAPE_AOPEN << 61) | tc;
                    tape[tc++] = JBIN_TAPE_MAKE_CLOSE(0, open_idx);
                    goto have_value;
                }
                /* Tight inner loop for arrays of literals (true/false/null) */
                if (fc == 't' || fc == 'f' || fc == 'n') {
                    for (;;) {
                        uint32_t vpos = si_next(&it);
                        uint8_t vc = input[vpos];
                        if (vc == 't') {
                            uint32_t w; __builtin_memcpy(&w, input + vpos, 4);
                            if (UNLIKELY(w != 0x65757274u)) {
                                p->cur = input + vpos;
                                fail(p, JBIN_ERR_BAD_LITERAL);
                                return JBIN_NONE;
                            }
                            tape[tc++] = JBIN_TAPE_MAKE_LIT(JBIN_TAPE_TRUE);
                        } else if (vc == 'f') {
                            uint32_t w; __builtin_memcpy(&w, input + vpos + 1, 4);
                            if (UNLIKELY(w != 0x65736c61u)) {
                                p->cur = input + vpos;
                                fail(p, JBIN_ERR_BAD_LITERAL);
                                return JBIN_NONE;
                            }
                            tape[tc++] = JBIN_TAPE_MAKE_LIT(JBIN_TAPE_FALSE);
                        } else if (vc == 'n') {
                            uint32_t w; __builtin_memcpy(&w, input + vpos, 4);
                            if (UNLIKELY(w != 0x6c6c756eu)) {
                                p->cur = input + vpos;
                                fail(p, JBIN_ERR_BAD_LITERAL);
                                return JBIN_NONE;
                            }
                            tape[tc++] = JBIN_TAPE_MAKE_LIT(JBIN_TAPE_NULL);
                        } else {
                            /* Non-literal: push frame, fall to generic dispatch */
                            pos = vpos;
                            c = vc;
                            sp++;
                            stack[sp].open_idx = open_idx;
                            stack[sp].is_obj = 0;
                            goto do_dispatch;
                        }
                        if (UNLIKELY(!si_has(&it))) {
                            p->error = JBIN_ERR_UNEXPECTED;
                            p->error_pos = vpos;
                            return JBIN_NONE;
                        }
                        if (UNLIKELY(tc + 3 >= JBIN_MAX_TAPE)) {
                            p->error = JBIN_ERR_NODES_FULL;
                            p->error_pos = vpos;
                            return JBIN_NONE;
                        }
                        uint32_t npos = si_next(&it);
                        if (input[npos] == ']') {
                            last_pos = npos;
                            tape[open_idx] = ((uint64_t)JBIN_TAPE_AOPEN << 61) | tc;
                            tape[tc++] = JBIN_TAPE_MAKE_CLOSE(0, open_idx);
                            goto have_value;
                        }
                        /* npos is comma — continue */
                    }
                }
            }
            sp++;
            stack[sp].open_idx = open_idx;
            stack[sp].is_obj = 0;
            goto do_value;
        }
        if (c == '{') {
            last_pos = pos;
            if (UNLIKELY(sp >= JBIN_MAX_DEPTH - 1)) {
                p->cur = input + pos;
                fail(p, JBIN_ERR_DEPTH);
                return JBIN_NONE;
            }
            uint32_t open_idx = tc;
            tape[tc++] = JBIN_TAPE_MAKE_OPEN(JBIN_TAPE_OOPEN);
            if (UNLIKELY(!si_has(&it))) {
                p->error = JBIN_ERR_UNEXPECTED;
                p->error_pos = pos;
                return JBIN_NONE;
            }
            {
                uint32_t npos = si_next(&it);
                uint8_t fc = input[npos];
                if (fc == '}') {
                    last_pos = npos;
                    tape[open_idx] = ((uint64_t)JBIN_TAPE_OOPEN << 61) | tc;
                    tape[tc++] = JBIN_TAPE_MAKE_CLOSE(1, open_idx);
                    goto have_value;
                }
                if (UNLIKELY(fc != '"')) {
                    p->error = JBIN_ERR_EXPECTED_KEY;
                    p->error_pos = npos;
                    return JBIN_NONE;
                }
                uint32_t kclose = si_next(&it);
                uint32_t klen = kclose - npos - 1;
                if (all_clean || LIKELY(!(dirty[npos >> 6] | dirty[kclose >> 6]))) {
                    tape[tc++] = JBIN_TAPE_MAKE_STR(1, npos + 1, klen);
                } else {
                    int fast = tape_string_inline(p, input, tape, &tc, npos, klen,
                                                      vbs_s2, vxor_s2, vthresh_s2);
                    if (fast == 0) {
                        p->cur = input + npos;
                        if (UNLIKELY(!tape_string_fused(p, tape, &tc, klen)))
                            return JBIN_NONE;
                    } else if (UNLIKELY(fast < 0))
                        return JBIN_NONE;
                }
            }
            {
                uint32_t cpos = si_next(&it);
                if (UNLIKELY(input[cpos] != ':')) {
                    p->error = JBIN_ERR_EXPECTED_COLON;
                    p->error_pos = cpos;
                    return JBIN_NONE;
                }
            }
            sp++;
            stack[sp].open_idx = open_idx;
            stack[sp].is_obj = 1;
            goto do_value;
        }
    }

    p->error = JBIN_ERR_UNEXPECTED;
    p->error_pos = si_pos(&it);
    return JBIN_NONE;

have_value:
    if (sp < 0) {
        if (UNLIKELY(si_has(&it))) {
            p->error = JBIN_ERR_TRAILING;
            p->error_pos = si_pos(&it);
            return JBIN_NONE;
        }
        p->node_count = tc;
        return 0;
    }

    if (stack[sp].is_obj) {
        if (UNLIKELY(!si_has(&it))) {
            p->error = JBIN_ERR_UNEXPECTED;
            p->error_pos = last_pos;
            return JBIN_NONE;
        }
        {
            uint32_t npos = si_next(&it);
            if (input[npos] == '}') {
                last_pos = npos;
                uint32_t oi = stack[sp].open_idx;
                tape[oi] = ((uint64_t)JBIN_TAPE_OOPEN << 61) | tc;
                tape[tc++] = JBIN_TAPE_MAKE_CLOSE(1, oi);
                sp--;
                goto have_value;
            }
            if (UNLIKELY(input[npos] != ',')) {
                p->error = JBIN_ERR_UNEXPECTED;
                p->error_pos = npos;
                return JBIN_NONE;
            }
        }

        {
            uint32_t npos = si_next(&it);
            if (UNLIKELY(input[npos] != '"')) {
                p->error = JBIN_ERR_EXPECTED_KEY;
                p->error_pos = npos;
                return JBIN_NONE;
            }
            uint32_t kclose = si_next(&it);
            uint32_t klen = kclose - npos - 1;
            if (all_clean || LIKELY(!(dirty[npos >> 6] | dirty[kclose >> 6]))) {
                tape[tc++] = JBIN_TAPE_MAKE_STR(1, npos + 1, klen);
            } else {
                int fast = tape_string_inline(p, input, tape, &tc, npos, klen,
                                                  vbs_s2, vxor_s2, vthresh_s2);
                if (fast == 0) {
                    p->cur = input + npos;
                    if (UNLIKELY(!tape_string_fused(p, tape, &tc, klen)))
                        return JBIN_NONE;
                } else if (UNLIKELY(fast < 0))
                    return JBIN_NONE;
            }
        }

        {
            uint32_t cpos = si_next(&it);
            if (UNLIKELY(input[cpos] != ':')) {
                p->error = JBIN_ERR_EXPECTED_COLON;
                p->error_pos = cpos;
                return JBIN_NONE;
            }
        }

        /* Fused: fetch next value structural without going through do_value */
        if (UNLIKELY(!si_has(&it))) {
            p->error = JBIN_ERR_UNEXPECTED;
            p->error_pos = last_pos;
            return JBIN_NONE;
        }
        if (UNLIKELY(tc + 3 >= JBIN_MAX_TAPE)) {
            p->error = JBIN_ERR_NODES_FULL;
            p->error_pos = last_pos;
            return JBIN_NONE;
        }
        pos = si_next(&it);
        c = input[pos];
        goto do_dispatch;
    }

    if (UNLIKELY(!si_has(&it))) {
        p->error = JBIN_ERR_UNEXPECTED;
        p->error_pos = last_pos;
        return JBIN_NONE;
    }
    {
        uint32_t npos = si_next(&it);
        if (input[npos] == ']') {
            last_pos = npos;
            uint32_t oi = stack[sp].open_idx;
            tape[oi] = ((uint64_t)JBIN_TAPE_AOPEN << 61) | tc;
            tape[tc++] = JBIN_TAPE_MAKE_CLOSE(0, oi);
            sp--;
            goto have_value;
        }
        if (UNLIKELY(input[npos] != ',')) {
            p->error = JBIN_ERR_UNEXPECTED;
            p->error_pos = npos;
            return JBIN_NONE;
        }
    }
    /* Fused: fetch next value structural without going through do_value */
    if (UNLIKELY(!si_has(&it))) {
        p->error = JBIN_ERR_UNEXPECTED;
        p->error_pos = last_pos;
        return JBIN_NONE;
    }
    if (UNLIKELY(tc + 3 >= JBIN_MAX_TAPE)) {
        p->error = JBIN_ERR_NODES_FULL;
        p->error_pos = last_pos;
        return JBIN_NONE;
    }
    pos = si_next(&it);
    c = input[pos];
    goto do_dispatch;
}

static uint32_t stage2_tape_bitmask(P *p, SiIter *itp, const uint8_t *dirty) {
    return stage2_tape_core(p, itp, dirty, 0);
}

static uint32_t stage2_tape_clean(P *p, SiIter *itp) {
    return stage2_tape_core(p, itp, NULL, 1);
}

#endif /* JBIN_TWOPASS */

#if 0 /* flat-index stage2: experimentally slower than SiIter */
FINLINE uint32_t stage2_tape_flat(P *p, const uint32_t *si,
                                   const uint32_t *si_end,
                                   const uint8_t *dirty,
                                   const int all_clean) {
    TapeFrame stack[JBIN_MAX_DEPTH];
    int sp = -1;
    uint64_t *tape = p->arena->tape;
    const uint8_t *input = p->input;

    uint32_t last_pos = 0;
    uint32_t tc = 0;
    uint32_t pos;
    uint8_t c;

do_value:
    if (UNLIKELY(si >= si_end)) {
        p->error = JBIN_ERR_UNEXPECTED;
        p->error_pos = last_pos;
        return JBIN_NONE;
    }
    if (UNLIKELY(tc + 3 >= JBIN_MAX_TAPE)) {
        p->error = JBIN_ERR_NODES_FULL;
        p->error_pos = last_pos;
        return JBIN_NONE;
    }
    pos = *si++;
    c = input[pos];
do_dispatch: ;
    {
        __builtin_prefetch(input + si[8], 0, 1);
        if (c == '"') {
            uint32_t close_pos = *si++;
            uint32_t slen = close_pos - pos - 1;
            if (all_clean || LIKELY(!(dirty[pos >> 6] | dirty[close_pos >> 6]))) {
                tape[tc++] = JBIN_TAPE_MAKE_STR(1, pos + 1, slen);
            } else {
                p->cur = input + pos;
                if (UNLIKELY(!tape_string_fused(p, tape, &tc)))
                    return JBIN_NONE;
            }
            last_pos = pos;
            goto have_value;
        }
        if (c == '-' || (uint32_t)(c - '0') <= 9u) {
            if (LIKELY(si < si_end)) {
                if (UNLIKELY(!parse_number_s2_tape(p, input + pos, tape, &tc)))
                    return JBIN_NONE;
            } else {
                p->cur = input + pos;
                if (UNLIKELY(!parse_number_tape_bounded(p, tape, &tc)))
                    return JBIN_NONE;
            }
            last_pos = pos;
            goto have_value;
        }
        if (c == 't') {
            uint32_t w; __builtin_memcpy(&w, input + pos, 4);
            if (LIKELY(w == 0x65757274u)) {
                tape[tc++] = JBIN_TAPE_MAKE_LIT(JBIN_TAPE_TRUE);
                last_pos = pos;
                goto have_value;
            }
            p->cur = input + pos;
            fail(p, JBIN_ERR_BAD_LITERAL);
            return JBIN_NONE;
        }
        if (c == 'f') {
            uint32_t w; __builtin_memcpy(&w, input + pos + 1, 4);
            if (LIKELY(w == 0x65736c61u)) {
                tape[tc++] = JBIN_TAPE_MAKE_LIT(JBIN_TAPE_FALSE);
                last_pos = pos;
                goto have_value;
            }
            p->cur = input + pos;
            fail(p, JBIN_ERR_BAD_LITERAL);
            return JBIN_NONE;
        }
        if (c == 'n') {
            uint32_t w; __builtin_memcpy(&w, input + pos, 4);
            if (LIKELY(w == 0x6c6c756eu)) {
                tape[tc++] = JBIN_TAPE_MAKE_LIT(JBIN_TAPE_NULL);
                last_pos = pos;
                goto have_value;
            }
            p->cur = input + pos;
            fail(p, JBIN_ERR_BAD_LITERAL);
            return JBIN_NONE;
        }
        if (c == '[') {
            last_pos = pos;
            if (UNLIKELY(sp >= JBIN_MAX_DEPTH - 1)) {
                p->cur = input + pos;
                fail(p, JBIN_ERR_DEPTH);
                return JBIN_NONE;
            }
            uint32_t open_idx = tc;
            tape[tc++] = JBIN_TAPE_MAKE_OPEN(JBIN_TAPE_AOPEN);
            if (UNLIKELY(si >= si_end)) {
                p->error = JBIN_ERR_UNEXPECTED;
                p->error_pos = pos;
                return JBIN_NONE;
            }
            {
                uint32_t fpos = *si;
                if (input[fpos] == ']') {
                    last_pos = fpos;
                    si++;
                    tape[open_idx] = ((uint64_t)JBIN_TAPE_AOPEN << 61) | tc;
                    tape[tc++] = JBIN_TAPE_MAKE_CLOSE(0, open_idx);
                    goto have_value;
                }
            }
            sp++;
            stack[sp].open_idx = open_idx;
            stack[sp].is_obj = 0;
            goto do_value;
        }
        if (c == '{') {
            last_pos = pos;
            if (UNLIKELY(sp >= JBIN_MAX_DEPTH - 1)) {
                p->cur = input + pos;
                fail(p, JBIN_ERR_DEPTH);
                return JBIN_NONE;
            }
            uint32_t open_idx = tc;
            tape[tc++] = JBIN_TAPE_MAKE_OPEN(JBIN_TAPE_OOPEN);
            if (UNLIKELY(si >= si_end)) {
                p->error = JBIN_ERR_UNEXPECTED;
                p->error_pos = pos;
                return JBIN_NONE;
            }
            {
                uint32_t npos = *si++;
                uint8_t fc = input[npos];
                if (fc == '}') {
                    last_pos = npos;
                    tape[open_idx] = ((uint64_t)JBIN_TAPE_OOPEN << 61) | tc;
                    tape[tc++] = JBIN_TAPE_MAKE_CLOSE(1, open_idx);
                    goto have_value;
                }
                if (UNLIKELY(fc != '"')) {
                    p->error = JBIN_ERR_EXPECTED_KEY;
                    p->error_pos = npos;
                    return JBIN_NONE;
                }
                uint32_t kclose = *si++;
                uint32_t klen = kclose - npos - 1;
                if (all_clean || LIKELY(!(dirty[npos >> 6] | dirty[kclose >> 6]))) {
                    tape[tc++] = JBIN_TAPE_MAKE_STR(1, npos + 1, klen);
                } else {
                    p->cur = input + npos;
                    if (UNLIKELY(!tape_string_fused(p, tape, &tc)))
                        return JBIN_NONE;
                }
            }
            {
                uint32_t cpos = *si++;
                if (UNLIKELY(input[cpos] != ':')) {
                    p->error = JBIN_ERR_EXPECTED_COLON;
                    p->error_pos = cpos;
                    return JBIN_NONE;
                }
            }
            sp++;
            stack[sp].open_idx = open_idx;
            stack[sp].is_obj = 1;
            goto do_value;
        }
    }

    p->error = JBIN_ERR_UNEXPECTED;
    p->error_pos = (si < si_end) ? *si : last_pos;
    return JBIN_NONE;

have_value:
    if (sp < 0) {
        if (UNLIKELY(si < si_end)) {
            p->error = JBIN_ERR_TRAILING;
            p->error_pos = *si;
            return JBIN_NONE;
        }
        p->node_count = tc;
        return 0;
    }

    if (stack[sp].is_obj) {
        if (UNLIKELY(si >= si_end)) {
            p->error = JBIN_ERR_UNEXPECTED;
            p->error_pos = last_pos;
            return JBIN_NONE;
        }
        {
            uint32_t npos = *si++;
            if (input[npos] == '}') {
                last_pos = npos;
                uint32_t oi = stack[sp].open_idx;
                tape[oi] = ((uint64_t)JBIN_TAPE_OOPEN << 61) | tc;
                tape[tc++] = JBIN_TAPE_MAKE_CLOSE(1, oi);
                sp--;
                goto have_value;
            }
            if (UNLIKELY(input[npos] != ',')) {
                p->error = JBIN_ERR_UNEXPECTED;
                p->error_pos = npos;
                return JBIN_NONE;
            }
        }

        {
            uint32_t npos = *si++;
            if (UNLIKELY(input[npos] != '"')) {
                p->error = JBIN_ERR_EXPECTED_KEY;
                p->error_pos = npos;
                return JBIN_NONE;
            }
            uint32_t kclose = *si++;
            uint32_t klen = kclose - npos - 1;
            if (all_clean || LIKELY(!(dirty[npos >> 6] | dirty[kclose >> 6]))) {
                tape[tc++] = JBIN_TAPE_MAKE_STR(1, npos + 1, klen);
            } else {
                p->cur = input + npos;
                if (UNLIKELY(!tape_string_fused(p, tape, &tc)))
                    return JBIN_NONE;
            }
        }

        {
            uint32_t cpos = *si++;
            if (UNLIKELY(input[cpos] != ':')) {
                p->error = JBIN_ERR_EXPECTED_COLON;
                p->error_pos = cpos;
                return JBIN_NONE;
            }
        }

        if (UNLIKELY(si >= si_end)) {
            p->error = JBIN_ERR_UNEXPECTED;
            p->error_pos = last_pos;
            return JBIN_NONE;
        }
        if (UNLIKELY(tc + 3 >= JBIN_MAX_TAPE)) {
            p->error = JBIN_ERR_NODES_FULL;
            p->error_pos = last_pos;
            return JBIN_NONE;
        }
        pos = *si++;
        c = input[pos];
        goto do_dispatch;
    }

    if (UNLIKELY(si >= si_end)) {
        p->error = JBIN_ERR_UNEXPECTED;
        p->error_pos = last_pos;
        return JBIN_NONE;
    }
    {
        uint32_t npos = *si++;
        if (input[npos] == ']') {
            last_pos = npos;
            uint32_t oi = stack[sp].open_idx;
            tape[oi] = ((uint64_t)JBIN_TAPE_AOPEN << 61) | tc;
            tape[tc++] = JBIN_TAPE_MAKE_CLOSE(0, oi);
            sp--;
            goto have_value;
        }
        if (UNLIKELY(input[npos] != ',')) {
            p->error = JBIN_ERR_UNEXPECTED;
            p->error_pos = npos;
            return JBIN_NONE;
        }
    }
    if (UNLIKELY(si >= si_end)) {
        p->error = JBIN_ERR_UNEXPECTED;
        p->error_pos = last_pos;
        return JBIN_NONE;
    }
    if (UNLIKELY(tc + 3 >= JBIN_MAX_TAPE)) {
        p->error = JBIN_ERR_NODES_FULL;
        p->error_pos = last_pos;
        return JBIN_NONE;
    }
    pos = *si++;
    c = input[pos];
    goto do_dispatch;
}

static uint32_t stage2_tape_flat_dirty(P *p, const uint32_t *si,
                                        const uint32_t *si_end,
                                        const uint8_t *dirty) {
    return stage2_tape_flat(p, si, si_end, dirty, 0);
}

static uint32_t stage2_tape_flat_clean(P *p, const uint32_t *si,
                                        const uint32_t *si_end) {
    return stage2_tape_flat(p, si, si_end, NULL, 1);
}
#endif /* flat-index stage2 */

static uint32_t parse_root(P *p) {
    Frame stack[JBIN_MAX_DEPTH];
    int sp = -1;
    uint32_t depth = 0;
    uint32_t value;
    uint8_t c;
    JbinNode *nodes = p->arena->nodes;

do_value:
    if (UNLIKELY(p->cur >= p->end)) {
        fail(p, JBIN_ERR_UNEXPECTED);
        return JBIN_NONE;
    }

    c = *p->cur;

    if (c == '"') {
        value = parse_string_fast(p);
        if (UNLIKELY(value == JBIN_NONE)) return JBIN_NONE;
        goto have_value;
    }
    if (c == '-' || (uint32_t)(c - '0') <= 9u) {
        value = parse_number(p);
        if (UNLIKELY(value == JBIN_NONE)) return JBIN_NONE;
        goto have_value;
    }
    if (c == 't') {
        uint32_t w; __builtin_memcpy(&w, p->cur, 4);
        if (LIKELY(p->cur + 4 <= p->end && w == 0x65757274u)) {
            p->cur += 4;
            value = alloc_node(p, JBIN_TRUE);
            if (UNLIKELY(value == JBIN_NONE)) return JBIN_NONE;
            goto have_value;
        }
        fail(p, JBIN_ERR_BAD_LITERAL);
        return JBIN_NONE;
    }
    if (c == 'f') {
        uint32_t w; __builtin_memcpy(&w, p->cur + 1, 4);
        if (LIKELY(p->cur + 5 <= p->end && w == 0x65736c61u)) {
            p->cur += 5;
            value = alloc_node(p, JBIN_FALSE);
            if (UNLIKELY(value == JBIN_NONE)) return JBIN_NONE;
            goto have_value;
        }
        fail(p, JBIN_ERR_BAD_LITERAL);
        return JBIN_NONE;
    }
    if (c == 'n') {
        uint32_t w; __builtin_memcpy(&w, p->cur, 4);
        if (LIKELY(p->cur + 4 <= p->end && w == 0x6c6c756eu)) {
            p->cur += 4;
            value = alloc_node(p, JBIN_NULL);
            if (UNLIKELY(value == JBIN_NONE)) return JBIN_NONE;
            goto have_value;
        }
        fail(p, JBIN_ERR_BAD_LITERAL);
        return JBIN_NONE;
    }
    if (c == '[') {
        p->cur++;
        depth++;
        if (UNLIKELY(depth > JBIN_MAX_DEPTH)) {
            fail(p, JBIN_ERR_DEPTH);
            return JBIN_NONE;
        }
        value = alloc_node(p, JBIN_ARRAY);
        if (UNLIKELY(value == JBIN_NONE)) return JBIN_NONE;
        nodes[value].first_child = JBIN_NONE;
        skip_ws(p);
        if (p->cur < p->end && *p->cur == ']') {
            p->cur++;
            depth--;
            goto have_value;
        }
        sp++;
        stack[sp].node = value;
        stack[sp].link = &nodes[value].first_child;
        stack[sp].is_obj = 0;
        goto do_value;
    }
    if (c == '{') {
        p->cur++;
        depth++;
        if (UNLIKELY(depth > JBIN_MAX_DEPTH)) {
            fail(p, JBIN_ERR_DEPTH);
            return JBIN_NONE;
        }
        uint32_t obj = alloc_node(p, JBIN_OBJECT);
        if (UNLIKELY(obj == JBIN_NONE)) return JBIN_NONE;
        nodes[obj].first_child = JBIN_NONE;
        skip_ws(p);
        if (p->cur < p->end && *p->cur == '}') {
            p->cur++;
            depth--;
            value = obj;
            goto have_value;
        }
        if (UNLIKELY(p->cur >= p->end || *p->cur != '"')) {
            fail(p, JBIN_ERR_EXPECTED_KEY);
            return JBIN_NONE;
        }
        uint32_t key = parse_string_fast(p);
        if (UNLIKELY(key == JBIN_NONE)) return JBIN_NONE;
        skip_ws(p);
        if (UNLIKELY(p->cur >= p->end || *p->cur != ':')) {
            fail(p, JBIN_ERR_EXPECTED_COLON);
            return JBIN_NONE;
        }
        p->cur++;
        skip_ws(p);
        sp++;
        stack[sp].node = obj;
        stack[sp].link = &nodes[obj].first_child;
        stack[sp].is_obj = 1;
        stack[sp].key = key;
        goto do_value;
    }

    fail(p, JBIN_ERR_UNEXPECTED);
    return JBIN_NONE;

have_value:
    if (sp < 0) {
        jbin_node_set(&nodes[value], jbin_node_type(&nodes[value]), JBIN_NONE);
        return value;
    }

    if (stack[sp].is_obj) {
        uint32_t key = stack[sp].key;
        jbin_node_set(&nodes[key], jbin_node_type(&nodes[key]), value);
        LINK_SET(stack[sp].link, key);
        stack[sp].link = &nodes[value].type_next;

        skip_ws(p);
        if (UNLIKELY(p->cur >= p->end)) {
            fail(p, JBIN_ERR_UNEXPECTED);
            return JBIN_NONE;
        }
        c = *p->cur;
        if (c == '}') {
            p->cur++;
            depth--;
            LINK_SET(stack[sp].link, JBIN_NONE);
            value = stack[sp].node;
            sp--;
            goto have_value;
        }
        if (UNLIKELY(c != ',')) {
            fail(p, JBIN_ERR_UNEXPECTED);
            return JBIN_NONE;
        }
        p->cur++;
        skip_ws(p);
        if (UNLIKELY(p->cur >= p->end || *p->cur != '"')) {
            fail(p, JBIN_ERR_EXPECTED_KEY);
            return JBIN_NONE;
        }
        key = parse_string_fast(p);
        if (UNLIKELY(key == JBIN_NONE)) return JBIN_NONE;
        skip_ws(p);
        if (UNLIKELY(p->cur >= p->end || *p->cur != ':')) {
            fail(p, JBIN_ERR_EXPECTED_COLON);
            return JBIN_NONE;
        }
        p->cur++;
        skip_ws(p);
        stack[sp].key = key;
        goto do_value;
    }

    LINK_SET(stack[sp].link, value);
    stack[sp].link = &nodes[value].type_next;

    skip_ws(p);
    if (UNLIKELY(p->cur >= p->end)) {
        fail(p, JBIN_ERR_UNEXPECTED);
        return JBIN_NONE;
    }
    c = *p->cur;
    if (c == ']') {
        p->cur++;
        depth--;
        LINK_SET(stack[sp].link, JBIN_NONE);
        value = stack[sp].node;
        sp--;
        goto have_value;
    }
    if (UNLIKELY(c != ',')) {
        fail(p, JBIN_ERR_UNEXPECTED);
        return JBIN_NONE;
    }
    p->cur++;
    skip_ws(p);
    goto do_value;
}

void jbin_arena_init(JbinArena *arena) {
    arena->node_count = 0;
    arena->string_used = 0;
}

JbinResult jbin_parse(JbinArena *arena, const char *input, uint32_t length) {
    P p;
    p.input     = (const uint8_t *)input;
    p.cur       = (const uint8_t *)input;
    p.end       = (const uint8_t *)input + length;
    p.arena     = arena;
    p.error     = JBIN_OK;
    p.error_pos = 0;

    arena->node_count = 0;
    arena->string_used = 0;
    arena->is_tape = 0;
    p.node_count = 0;

#ifdef JBIN_TWOPASS
    if (LIKELY(length >= 64)) {
        uint32_t sample_off = length / 4;
        uint32_t numeric = 0, ws = 0, quotes = 0;
        for (uint32_t i = 0; i < 64; i++) {
            uint8_t c = p.input[sample_off + i];
            numeric += ((uint32_t)(c - '0') <= 9u) | (c == '.') | (c == '-');
            ws += (c == ' ') | (c == '\n') | (c == '\r') | (c == '\t');
            quotes += (c == '"');
        }

        if ((numeric < 56 || length > (8u * 1024u * 1024u)) && ws < 48) {
            uint32_t max_blocks = (length + 63) / 64;
            uint64_t *bm_out = (uint64_t *)arena->structural;
            uint8_t *dirty = (uint8_t *)(bm_out + max_blocks);
            uint8_t any_dirty = 0;
            uint32_t num_blocks = stage1_bitmask(p.input, length,
                                                  bm_out, max_blocks,
                                                  dirty,
                                                  &any_dirty);
            if (UNLIKELY(num_blocks == 0)) {
                JbinResult r = { JBIN_ERR_EMPTY, 0, JBIN_NONE };
                return r;
            }

            SiIter it;
            si_init(&it, bm_out, num_blocks);
            uint32_t root;
            arena->is_tape = 1;
            root = any_dirty
                ? stage2_tape_bitmask(&p, &it, dirty)
                : stage2_tape_clean(&p, &it);
            arena->node_count = p.node_count;
            if (UNLIKELY(root == JBIN_NONE)) {
                JbinResult r = { p.error, p.error_pos, JBIN_NONE };
                return r;
            }
            JbinResult r = { JBIN_OK, 0, root };
            return r;
        }
    }
#endif

    skip_ws(&p);

    if (UNLIKELY(p.cur >= p.end)) {
        JbinResult r = { JBIN_ERR_EMPTY, 0, JBIN_NONE };
        return r;
    }

    uint32_t root = parse_root(&p);
    arena->node_count = p.node_count;
    if (UNLIKELY(root == JBIN_NONE)) {
        JbinResult r = { p.error, p.error_pos, JBIN_NONE };
        return r;
    }

    skip_ws(&p);

    if (UNLIKELY(p.cur < p.end)) {
        JbinResult r = { JBIN_ERR_TRAILING, POS(&p), JBIN_NONE };
        return r;
    }

    JbinResult r = { JBIN_OK, 0, root };
    return r;
}

const char *jbin_error_str(JbinError err) {
    switch (err) {
        case JBIN_OK:                      return "ok";
        case JBIN_ERR_EMPTY:               return "empty input";
        case JBIN_ERR_UNEXPECTED:          return "unexpected character";
        case JBIN_ERR_TRAILING:            return "trailing content";
        case JBIN_ERR_DEPTH:               return "max depth exceeded";
        case JBIN_ERR_NODES_FULL:          return "node pool exhausted";
        case JBIN_ERR_STRING_FULL:         return "string buffer exhausted";
        case JBIN_ERR_UNTERMINATED_STRING: return "unterminated string";
        case JBIN_ERR_BAD_ESCAPE:          return "invalid escape sequence";
        case JBIN_ERR_BAD_UNICODE:         return "invalid unicode escape";
        case JBIN_ERR_BAD_UTF8:            return "invalid utf-8";
        case JBIN_ERR_BAD_NUMBER:          return "invalid number";
        case JBIN_ERR_BAD_LITERAL:         return "invalid literal";
        case JBIN_ERR_EXPECTED_COLON:      return "expected ':'";
        case JBIN_ERR_EXPECTED_KEY:        return "expected string key";
        case JBIN_ERR_CONTROL_CHAR:        return "unescaped control character";
    }
    return "unknown error";
}

const char *jbin_type_str(JbinType type) {
    switch (type) {
        case JBIN_NULL:   return "null";
        case JBIN_TRUE:   return "true";
        case JBIN_FALSE:  return "false";
        case JBIN_NUMBER: return "number";
        case JBIN_STRING: return "string";
        case JBIN_ARRAY:  return "array";
        case JBIN_OBJECT: return "object";
    }
    return "unknown";
}
