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

#define JBIN_NONE      ((uint32_t)0x1FFFFFFF)
#define JBIN_INPUT_REF ((uint32_t)0x80000000)

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
    JbinNode nodes[JBIN_MAX_NODES];
    char     strings[JBIN_MAX_STRING];
    uint32_t structural[JBIN_MAX_STRUCTURAL];
    uint32_t node_count;
    uint32_t string_used;
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

#endif
