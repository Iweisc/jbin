#ifndef JBIN_H
#define JBIN_H

#include <stdint.h>

#ifndef JBIN_MAX_NODES
#define JBIN_MAX_NODES 4096
#endif

#ifndef JBIN_MAX_DEPTH
#define JBIN_MAX_DEPTH 256
#endif

#ifndef JBIN_MAX_STRING
#define JBIN_MAX_STRING (256u * 1024u)
#endif

#ifndef JBIN_MAX_STRUCTURAL
#define JBIN_MAX_STRUCTURAL (JBIN_MAX_NODES * 2)
#endif

#define JBIN_MAX_TAPE  ((JBIN_MAX_NODES * 12u) / 8u)

#define JBIN_NONE      ((uint32_t)0x1FFFFFFF)
#define JBIN_INPUT_REF ((uint32_t)0x80000000)

/* Tape entry encoding (64 bits):
 *   Bits 63-61: type tag
 *   STRING/NUMBER: bit 60=input_ref, bits 59-30=offset, bits 29-0=length
 *   OPEN:  bits 31-0 = matching close tape index (patched on close)
 *   CLOSE: bit 60=is_object, bits 31-0 = matching open tape index
 */
#define JBIN_TAPE_NULL   0
#define JBIN_TAPE_TRUE   1
#define JBIN_TAPE_FALSE  2
#define JBIN_TAPE_NUM    3
#define JBIN_TAPE_STR    4
#define JBIN_TAPE_AOPEN  5
#define JBIN_TAPE_OOPEN  6
#define JBIN_TAPE_CLOSE  7

#define JBIN_TAPE_TYPE(e)    ((uint32_t)((e) >> 61))
#define JBIN_TAPE_OFFSET(e)  ((uint32_t)(((e) >> 30) & 0x3FFFFFFFu))
#define JBIN_TAPE_LENGTH(e)  ((uint32_t)((e) & 0x3FFFFFFFu))
#define JBIN_TAPE_IREF(e)    (((e) >> 60) & 1)
#define JBIN_TAPE_SCOPE(e)   ((uint32_t)((e) & 0xFFFFFFFFu))
#define JBIN_TAPE_ISOBJ(e)   (((e) >> 60) & 1)

#define JBIN_TAPE_MAKE_STR(ref,off,len) \
    (((uint64_t)JBIN_TAPE_STR << 61) | ((uint64_t)(ref) << 60) | \
     ((uint64_t)(off) << 30) | (uint64_t)(len))
#define JBIN_TAPE_MAKE_NUM(ref,off,len) \
    (((uint64_t)JBIN_TAPE_NUM << 61) | ((uint64_t)(ref) << 60) | \
     ((uint64_t)(off) << 30) | (uint64_t)(len))
#define JBIN_TAPE_MAKE_LIT(type) \
    ((uint64_t)(type) << 61)
#define JBIN_TAPE_MAKE_OPEN(type) \
    ((uint64_t)(type) << 61)
#define JBIN_TAPE_MAKE_CLOSE(is_obj, open_idx) \
    (((uint64_t)JBIN_TAPE_CLOSE << 61) | ((uint64_t)(is_obj) << 60) | \
     (uint64_t)(open_idx))

typedef enum {
    JBIN_NULL,
    JBIN_TRUE,
    JBIN_FALSE,
    JBIN_NUMBER,
    JBIN_STRING,
    JBIN_ARRAY,
    JBIN_OBJECT
} JbinType;

typedef enum {
    JBIN_OK = 0,
    JBIN_ERR_EMPTY,
    JBIN_ERR_UNEXPECTED,
    JBIN_ERR_TRAILING,
    JBIN_ERR_DEPTH,
    JBIN_ERR_NODES_FULL,
    JBIN_ERR_STRING_FULL,
    JBIN_ERR_UNTERMINATED_STRING,
    JBIN_ERR_BAD_ESCAPE,
    JBIN_ERR_BAD_UNICODE,
    JBIN_ERR_BAD_UTF8,
    JBIN_ERR_BAD_NUMBER,
    JBIN_ERR_BAD_LITERAL,
    JBIN_ERR_EXPECTED_COLON,
    JBIN_ERR_EXPECTED_KEY,
    JBIN_ERR_CONTROL_CHAR
} JbinError;

typedef struct {
    union {
        uint32_t first_child;
        struct {
            uint32_t str_off;
            uint32_t str_len;
        };
    };
    uint32_t type_next;  /* bits 31-29 = type, bits 28-0 = next */
} JbinNode;

#define JBIN_TYPE_SHIFT 29
#define JBIN_NEXT_MASK  ((uint32_t)0x1FFFFFFFu)

static inline JbinType jbin_node_type(const JbinNode *n) {
    return (JbinType)(n->type_next >> JBIN_TYPE_SHIFT);
}
static inline uint32_t jbin_node_next(const JbinNode *n) {
    return n->type_next & JBIN_NEXT_MASK;
}
static inline void jbin_node_set(JbinNode *n, JbinType type, uint32_t next) {
    n->type_next = ((uint32_t)type << JBIN_TYPE_SHIFT) | (next & JBIN_NEXT_MASK);
}

typedef struct {
    union {
        JbinNode nodes[JBIN_MAX_NODES];
        uint64_t tape[JBIN_MAX_TAPE];
    };
    char     strings[JBIN_MAX_STRING];
    uint32_t structural[JBIN_MAX_STRUCTURAL];
    uint32_t node_count;   /* reused as tape_count in tape mode */
    uint32_t string_used;
    uint8_t  is_tape;      /* 0=DOM, 1=tape */
} JbinArena;

typedef struct {
    JbinError error;
    uint32_t  error_pos;
    uint32_t  root;
} JbinResult;

void        jbin_arena_init(JbinArena *arena);
JbinResult  jbin_parse(JbinArena *arena, const char *input, uint32_t length);
const char *jbin_error_str(JbinError err);
const char *jbin_type_str(JbinType type);

static inline const char *jbin_str(const JbinArena *a, const JbinNode *n,
                                   const char *input) {
    uint32_t off = n->str_off;
    if (off & JBIN_INPUT_REF)
        return input + (off & ~JBIN_INPUT_REF);
    return a->strings + off;
}

/* --- Unified navigation API (works for both DOM and tape) --- */

static inline JbinType jbin_type(const JbinArena *a, uint32_t idx) {
    if (!a->is_tape)
        return jbin_node_type(&a->nodes[idx]);
    return (JbinType)JBIN_TAPE_TYPE(a->tape[idx]);
}

static inline const char *jbin_get_str(const JbinArena *a, uint32_t idx,
                                       const char *input, uint32_t *len) {
    if (!a->is_tape) {
        const JbinNode *n = &a->nodes[idx];
        *len = n->str_len;
        return jbin_str(a, n, input);
    }
    uint64_t e = a->tape[idx];
    *len = JBIN_TAPE_LENGTH(e);
    uint32_t off = JBIN_TAPE_OFFSET(e);
    if (JBIN_TAPE_IREF(e))
        return input + off;
    return a->strings + off;
}

static inline uint32_t jbin_tape_skip(const JbinArena *a, uint32_t idx) {
    uint32_t tt = JBIN_TAPE_TYPE(a->tape[idx]);
    if (tt == JBIN_TAPE_AOPEN || tt == JBIN_TAPE_OOPEN)
        return JBIN_TAPE_SCOPE(a->tape[idx]) + 1;
    return idx + 1;
}

static inline uint32_t jbin_container_close(const JbinArena *a, uint32_t idx) {
    if (a->is_tape)
        return JBIN_TAPE_SCOPE(a->tape[idx]);
    return JBIN_NONE;
}

static inline uint32_t jbin_first_child(const JbinArena *a, uint32_t idx) {
    if (!a->is_tape)
        return a->nodes[idx].first_child;
    uint32_t close_idx = JBIN_TAPE_SCOPE(a->tape[idx]);
    if (idx + 1 >= close_idx)
        return JBIN_NONE;
    return idx + 1;
}

static inline uint32_t jbin_next_sibling(const JbinArena *a, uint32_t idx,
                                          uint32_t parent_close) {
    if (!a->is_tape)
        return jbin_node_next(&a->nodes[idx]);
    uint32_t next = jbin_tape_skip(a, idx);
    if (next >= parent_close)
        return JBIN_NONE;
    return next;
}

static inline uint32_t jbin_obj_value(const JbinArena *a, uint32_t key_idx) {
    if (!a->is_tape)
        return jbin_node_next(&a->nodes[key_idx]);
    return key_idx + 1;
}

static inline uint32_t jbin_obj_next_key(const JbinArena *a, uint32_t key_idx,
                                          uint32_t obj_close) {
    if (!a->is_tape) {
        uint32_t val = jbin_node_next(&a->nodes[key_idx]);
        if (val == JBIN_NONE) return JBIN_NONE;
        return jbin_node_next(&a->nodes[val]);
    }
    uint32_t val_idx = key_idx + 1;
    uint32_t next_key = jbin_tape_skip(a, val_idx);
    if (next_key >= obj_close)
        return JBIN_NONE;
    return next_key;
}

#endif
