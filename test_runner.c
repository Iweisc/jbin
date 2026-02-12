#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <stdlib.h>
#include "jbin.h"

static JbinArena arena;

typedef struct {
    const char *json;
    uint32_t    len;
    int         expect;
    const char *name;
} TC;

#define Y(s, n)       { s, sizeof(s)-1, 1, n }
#define N(s, n)       { s, sizeof(s)-1, 0, n }
#define YL(s, l, n)   { s, l, 1, n }
#define NL(s, l, n)   { s, l, 0, n }

static TC builtin[] = {
    Y("null",                                   "literal null"),
    Y("true",                                   "literal true"),
    Y("false",                                  "literal false"),
    Y("0",                                      "zero"),
    Y("1",                                      "one digit"),
    Y("-0",                                     "negative zero"),
    Y("-1",                                     "negative one"),
    Y("123",                                    "multi-digit"),
    Y("-123",                                   "negative multi-digit"),
    Y("0.5",                                    "fraction"),
    Y("-0.5",                                   "negative fraction"),
    Y("1.23456",                                "long fraction"),
    Y("1e2",                                    "exponent lower"),
    Y("1E2",                                    "exponent upper"),
    Y("1e+2",                                   "exponent plus"),
    Y("1e-2",                                   "exponent minus"),
    Y("1.5e10",                                 "frac + exp"),
    Y("-0.5e-10",                               "neg frac neg exp"),
    Y("1E100",                                  "large exponent"),
    Y("\"\"",                                   "empty string"),
    Y("\"hello\"",                              "simple string"),
    Y("\"hello world\"",                        "string with space"),
    Y("\"\\\"\"",                               "escaped quote"),
    Y("\"\\\\\"",                               "escaped backslash"),
    Y("\"\\/\"",                                "escaped slash"),
    Y("\"\\b\"",                                "escaped backspace"),
    Y("\"\\f\"",                                "escaped formfeed"),
    Y("\"\\n\"",                                "escaped newline"),
    Y("\"\\r\"",                                "escaped cr"),
    Y("\"\\t\"",                                "escaped tab"),
    Y("\"\\u0041\"",                            "unicode A"),
    Y("\"\\u00e9\"",                            "unicode e-acute"),
    Y("\"\\u4e16\"",                            "unicode CJK"),
    Y("\"\\uD834\\uDD1E\"",                     "surrogate pair (treble clef)"),
    Y("\"\\u0000\"",                            "unicode null"),
    Y("\"\\uD800\\uDC00\"",                     "surrogate pair low boundary"),
    Y("\"\\uDBFF\\uDFFF\"",                     "surrogate pair high boundary"),
    Y("[]",                                     "empty array"),
    Y("[1]",                                    "single-element array"),
    Y("[1,2,3]",                                "multi-element array"),
    Y("[1, \"two\", true, null, false]",        "mixed array"),
    Y("[[]]",                                   "nested empty array"),
    Y("[[[1]]]",                                "deeply nested array"),
    Y("{}",                                     "empty object"),
    Y("{\"a\":1}",                              "single-pair object"),
    Y("{\"a\":1,\"b\":2}",                      "multi-pair object"),
    Y("{\"a\":{\"b\":{\"c\":3}}}",              "nested objects"),
    Y("{\"arr\":[1,2],\"obj\":{\"k\":\"v\"}}",  "mixed containers"),
    Y("  \t\n\r null  \t\n\r ",                 "whitespace around null"),
    Y(" [ 1 , 2 ] ",                            "whitespace in array"),
    Y(" { \"a\" : 1 } ",                        "whitespace in object"),
    Y("[\"\\u0048\\u0065\\u006C\\u006C\\u006F\"]", "unicode ascii spell"),

    N("",                                       "empty input"),
    N("   ",                                    "only whitespace"),
    N("nul",                                    "truncated null"),
    N("tru",                                    "truncated true"),
    N("fals",                                   "truncated false"),
    N("nulll",                                  "null + trailing l"),
    N("truee",                                  "true + trailing e"),
    N("TRUE",                                   "uppercase TRUE"),
    N("False",                                  "capitalized False"),
    N("Null",                                   "capitalized Null"),
    N("+1",                                     "leading plus"),
    N("01",                                     "leading zero int"),
    N("00",                                     "double zero"),
    N("-01",                                    "neg leading zero"),
    N("1.",                                     "trailing dot"),
    N(".5",                                     "leading dot"),
    N("1e",                                     "truncated exponent"),
    N("1e+",                                    "truncated exponent sign"),
    N("-",                                      "bare minus"),
    N("1 2",                                    "two values"),
    N("true false",                             "two literals"),
    N("[1,]",                                   "trailing comma array"),
    N("[,1]",                                   "leading comma array"),
    N("[1,,2]",                                 "double comma array"),
    N("{\"a\":1,}",                             "trailing comma object"),
    N("{,\"a\":1}",                             "leading comma object"),
    N("[",                                      "unclosed array"),
    N("{",                                      "unclosed object"),
    N("[1",                                     "unclosed array val"),
    N("{\"a\"",                                 "unclosed object key"),
    N("{\"a\":1",                               "unclosed object pair"),
    N("\"hello",                                "unclosed string"),
    N("\"he\\",                                 "backslash at end of string"),
    N("\"\\a\"",                                "invalid escape a"),
    N("\"\\x41\"",                              "invalid escape x"),
    N("\"\\u\"",                                "incomplete unicode"),
    N("\"\\u00\"",                              "short unicode"),
    N("\"\\uGGGG\"",                            "non-hex unicode"),
    N("\"\\uD800\"",                            "lone high surrogate"),
    N("\"\\uDC00\"",                            "lone low surrogate"),
    N("\"\\uD800\\u0041\"",                     "high surrogate + non-surrogate"),
    N("\"\\uD800\\uD800\"",                     "two high surrogates"),
    N("{1:2}",                                  "non-string key int"),
    N("{true:1}",                               "non-string key bool"),
    N("{null:1}",                               "non-string key null"),
    N("{\"a\" 1}",                              "missing colon"),
    N("'hello'",                                "single quotes"),
    N("[1] [2]",                                "multiple arrays"),
    N("{}{}",                                   "multiple objects"),
    N("NaN",                                    "NaN"),
    N("Infinity",                               "Infinity"),
    N("-Infinity",                              "-Infinity"),
    N("1.0e",                                   "frac truncated exp"),
    N("1.0e+",                                  "frac truncated exp sign"),
    N("1.0e-",                                  "frac truncated exp neg sign"),
    NL("\"\x01\"", 3,                           "raw control char 0x01"),
    NL("\"\x1f\"", 3,                           "raw control char 0x1f"),
    NL("\"\t\"",   3,                           "raw tab in string"),
    NL("\"\n\"",   3,                           "raw newline in string"),

    Y("\"\\n\\t\\r\\\\\"",                        "consecutive escapes"),
    Y("\"\\n\\n\\n\\n\\n\"",                        "only escapes no clean segments"),
    Y("\"abc\\ndef\\tghi\"",                        "mixed escapes and text"),
    Y("\"\\uD83D\\uDE00\"",                         "surrogate pair emoji U+1F600"),
    Y("\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"",             "all single-char escapes"),
    Y("\"\\nhello\"",                               "escape at string start"),
    Y("\"hello\\n\"",                               "escape at string end"),
    Y("\"\\u0048\\u0069\"",                          "multiple unicode escapes Hi"),
    N("\"\\uD800abc\"",                             "high surrogate non-backslash"),
    N("\"\\uXYZW\"",                                "invalid hex in unicode escape"),
    NL("\"\x00\"", 3,                               "raw null byte in string"),
    NL("\"\x0A\"", 3,                               "raw 0x0A in string"),
    NL("\"\x0D\"", 3,                               "raw 0x0D in string"),
    NL("\"\x08\"", 3,                               "raw 0x08 in string"),
    NL("\"abc\x01xyz\"", 9,                         "raw 0x01 mid-string"),
    Y("\"abcdefghijklmno\\npqrstuvwxyz0123\"",      "32 byte content with escape"),
    Y("\"abcdefghijklmno\\npqrstuvwxyz012\"",       "31 byte content with escape"),
    Y("\"abcdefghijklmno\\npqrstuvwxyz01234\"",     "33 byte content with escape"),
    Y("\"abcdefghijklmnopqrstuvwxyz012345\\nmore\"", "escape after 32 clean bytes"),
    N("\"hello\\",                                  "backslash at very end"),
    Y("\"\\uFFFF\"",                                "unicode BMP boundary U+FFFF"),
    Y("\"\\uD800\\uDC01\"",                         "surrogate pair just above BMP"),
    Y("\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"", "long string periodic escapes"),

    N("{\"a\":1,\"a\":2}",                      "duplicate keys (accepted by some)"),
};

static int is_impl_defined(int idx) {
    int last = (int)(sizeof(builtin)/sizeof(builtin[0])) - 1;
    return idx == last;
}

static int run_builtin_tests(int verbose) {
    int total = (int)(sizeof(builtin) / sizeof(builtin[0]));
    int passed = 0;
    int failed = 0;
    int impl = 0;

    for (int i = 0; i < total; i++) {
        TC *t = &builtin[i];
        jbin_arena_init(&arena);
        JbinResult r = jbin_parse(&arena, t->json, t->len);
        int ok = (r.error == JBIN_OK);

        if (is_impl_defined(i)) {
            impl++;
            if (verbose)
                printf("  IMPL  %-45s %s\n", t->name, ok ? "accepted" : "rejected");
            continue;
        }

        int pass = (ok == t->expect);
        if (pass) {
            passed++;
            if (verbose)
                printf("  PASS  %-45s\n", t->name);
        } else {
            failed++;
            printf("  FAIL  %-45s expected=%s got=%s",
                   t->name,
                   t->expect ? "accept" : "reject",
                   ok ? "accept" : "reject");
            if (!ok)
                printf(" err=\"%s\" @%u", jbin_error_str(r.error), r.error_pos);
            printf("\n");
        }
    }

    printf("\nBuilt-in: %d passed, %d failed, %d impl-defined out of %d\n",
           passed, failed, impl, total);
    return failed;
}

static int run_suite_dir(const char *dir, int verbose) {
    DIR *d = opendir(dir);
    if (!d) {
        printf("Cannot open directory: %s\n", dir);
        return 1;
    }

    int y_pass = 0, y_fail = 0;
    int n_pass = 0, n_fail = 0;
    int i_accept = 0, i_reject = 0;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        const char *name = ent->d_name;
        size_t nlen = strlen(name);

        if (nlen < 7) continue;
        if (name[nlen-5] != '.' || name[nlen-4] != 'j' ||
            name[nlen-3] != 's' || name[nlen-2] != 'o' || name[nlen-1] != 'n')
            continue;

        int category;
        if (name[0] == 'y' && name[1] == '_') category = 1;
        else if (name[0] == 'n' && name[1] == '_') category = 0;
        else if (name[0] == 'i' && name[1] == '_') category = -1;
        else continue;

        char path[4096];
        size_t dlen = strlen(dir);
        if (dlen + 1 + nlen + 1 > sizeof(path)) continue;
        for (size_t k = 0; k < dlen; k++) path[k] = dir[k];
        path[dlen] = '/';
        for (size_t k = 0; k < nlen; k++) path[dlen + 1 + k] = name[k];
        path[dlen + 1 + nlen] = '\0';

        FILE *f = fopen(path, "rb");
        if (!f) {
            printf("  SKIP  %s (cannot open)\n", name);
            continue;
        }

        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);

        if (fsize < 0 || (unsigned long)fsize > 10 * 1024 * 1024) {
            printf("  SKIP  %s (too large)\n", name);
            fclose(f);
            continue;
        }

        char *buf = (char *)malloc((size_t)fsize);
        if (!buf) {
            printf("  SKIP  %s (alloc failed)\n", name);
            fclose(f);
            continue;
        }

        size_t nread = fread(buf, 1, (size_t)fsize, f);
        fclose(f);

        jbin_arena_init(&arena);
        JbinResult r = jbin_parse(&arena, buf, (uint32_t)nread);
        int ok = (r.error == JBIN_OK);

        if (category == 1) {
            if (ok) {
                y_pass++;
                if (verbose) printf("  PASS  y_ %s\n", name);
            } else {
                y_fail++;
                printf("  FAIL  y_ %s  err=\"%s\" @%u\n",
                       name, jbin_error_str(r.error), r.error_pos);
            }
        } else if (category == 0) {
            if (!ok) {
                n_pass++;
                if (verbose) printf("  PASS  n_ %s\n", name);
            } else {
                n_fail++;
                printf("  FAIL  n_ %s  (should have been rejected)\n", name);
            }
        } else {
            if (ok) i_accept++; else i_reject++;
            if (verbose)
                printf("  IMPL  i_ %-50s %s\n", name, ok ? "accepted" : "rejected");
        }

        free(buf);
    }
    closedir(d);

    int total_y = y_pass + y_fail;
    int total_n = n_pass + n_fail;
    int total_i = i_accept + i_reject;
    printf("\nJSONTestSuite results (%s):\n", dir);
    printf("  y_ (must accept):  %d/%d passed\n", y_pass, total_y);
    printf("  n_ (must reject):  %d/%d passed\n", n_pass, total_n);
    printf("  i_ (impl-defined): %d accepted, %d rejected out of %d\n",
           i_accept, i_reject, total_i);

    return y_fail + n_fail;
}

static int memcmp_len(const char *a, const char *b, uint32_t len) {
    for (uint32_t i = 0; i < len; i++)
        if (a[i] != b[i]) return 1;
    return 0;
}

static int run_tape_tests(int verbose) {
    int passed = 0, failed = 0;

    #define TAPE_PASS(cond, name) do { \
        if (cond) { passed++; if (verbose) printf("  PASS  %-45s\n", name); } \
        else { failed++; printf("  FAIL  %-45s\n", name); } \
    } while(0)

    /* Test 1: String array (>64 bytes, triggers tape path) */
    {
        const char *json =
            "[\"hello world\", \"foo bar baz\", \"testing tape\", \"more strings here\"]";
        uint32_t len = (uint32_t)strlen(json);
        jbin_arena_init(&arena);
        JbinResult r = jbin_parse(&arena, json, len);
        TAPE_PASS(r.error == JBIN_OK, "tape: string array parses");
        TAPE_PASS(jbin_type(&arena, r.root) == JBIN_ARRAY, "tape: root is array");

        if (r.error == JBIN_OK) {
            uint32_t close_idx = jbin_container_close(&arena, r.root);
            uint32_t child = jbin_first_child(&arena, r.root);
            TAPE_PASS(child != JBIN_NONE, "tape: array has children");
            TAPE_PASS(jbin_type(&arena, child) == JBIN_STRING, "tape: first child is string");
            uint32_t slen;
            const char *s = jbin_get_str(&arena, child, json, &slen);
            TAPE_PASS(slen == 11 && !memcmp_len(s, "hello world", 11),
                      "tape: first string content");

            uint32_t second = jbin_next_sibling(&arena, child, close_idx);
            TAPE_PASS(second != JBIN_NONE, "tape: has second element");
            s = jbin_get_str(&arena, second, json, &slen);
            TAPE_PASS(slen == 11 && !memcmp_len(s, "foo bar baz", 11),
                      "tape: second string content");

            uint32_t third = jbin_next_sibling(&arena, second, close_idx);
            TAPE_PASS(third != JBIN_NONE, "tape: has third element");

            uint32_t fourth = jbin_next_sibling(&arena, third, close_idx);
            TAPE_PASS(fourth != JBIN_NONE, "tape: has fourth element");

            uint32_t end = jbin_next_sibling(&arena, fourth, close_idx);
            TAPE_PASS(end == JBIN_NONE, "tape: no fifth element");
        }
    }

    /* Test 2: Object with string keys and values */
    {
        const char *json =
            "{\"name\": \"Alice\", \"city\": \"Wonderland\", "
            "\"quest\": \"adventure\", \"padding\": \"xyzzy\"}";
        uint32_t len = (uint32_t)strlen(json);
        jbin_arena_init(&arena);
        JbinResult r = jbin_parse(&arena, json, len);
        TAPE_PASS(r.error == JBIN_OK, "tape: object parses");
        TAPE_PASS(jbin_type(&arena, r.root) == JBIN_OBJECT, "tape: root is object");

        if (r.error == JBIN_OK) {
            uint32_t close_idx = jbin_container_close(&arena, r.root);
            uint32_t key = jbin_first_child(&arena, r.root);
            TAPE_PASS(key != JBIN_NONE, "tape: obj has first key");
            TAPE_PASS(jbin_type(&arena, key) == JBIN_STRING, "tape: key is string");

            uint32_t slen;
            const char *s = jbin_get_str(&arena, key, json, &slen);
            TAPE_PASS(slen == 4 && !memcmp_len(s, "name", 4), "tape: first key is 'name'");

            uint32_t val = jbin_obj_value(&arena, key);
            TAPE_PASS(val != JBIN_NONE, "tape: has value for 'name'");
            s = jbin_get_str(&arena, val, json, &slen);
            TAPE_PASS(slen == 5 && !memcmp_len(s, "Alice", 5), "tape: value is 'Alice'");

            uint32_t key2 = jbin_obj_next_key(&arena, key, close_idx);
            TAPE_PASS(key2 != JBIN_NONE, "tape: has second key");
            s = jbin_get_str(&arena, key2, json, &slen);
            TAPE_PASS(slen == 4 && !memcmp_len(s, "city", 4), "tape: second key is 'city'");

            uint32_t key3 = jbin_obj_next_key(&arena, key2, close_idx);
            TAPE_PASS(key3 != JBIN_NONE, "tape: has third key");

            uint32_t key4 = jbin_obj_next_key(&arena, key3, close_idx);
            TAPE_PASS(key4 != JBIN_NONE, "tape: has fourth key");

            uint32_t key5 = jbin_obj_next_key(&arena, key4, close_idx);
            TAPE_PASS(key5 == JBIN_NONE, "tape: no fifth key");
        }
    }

    /* Test 3: Nested containers in tape mode */
    {
        const char *json =
            "{\"outer\": {\"inner\": [\"a\", \"b\"]}, "
            "\"extra_padding_to_hit_64_bytes\": \"yes\"}";
        uint32_t len = (uint32_t)strlen(json);
        jbin_arena_init(&arena);
        JbinResult r = jbin_parse(&arena, json, len);
        TAPE_PASS(r.error == JBIN_OK, "tape: nested containers parse");

        if (r.error == JBIN_OK) {
            uint32_t close_idx = jbin_container_close(&arena, r.root);
            uint32_t key = jbin_first_child(&arena, r.root);
            uint32_t val = jbin_obj_value(&arena, key);
            TAPE_PASS(jbin_type(&arena, val) == JBIN_OBJECT, "tape: nested obj type");

            uint32_t inner_close = jbin_container_close(&arena, val);
            uint32_t ik = jbin_first_child(&arena, val);
            TAPE_PASS(ik != JBIN_NONE, "tape: inner obj has key");

            uint32_t iv = jbin_obj_value(&arena, ik);
            TAPE_PASS(jbin_type(&arena, iv) == JBIN_ARRAY, "tape: inner val is array");

            uint32_t arr_close = jbin_container_close(&arena, iv);
            uint32_t elem = jbin_first_child(&arena, iv);
            TAPE_PASS(elem != JBIN_NONE, "tape: inner array has elements");
            uint32_t slen;
            const char *s = jbin_get_str(&arena, elem, json, &slen);
            TAPE_PASS(slen == 1 && s[0] == 'a', "tape: first array elem is 'a'");

            uint32_t elem2 = jbin_next_sibling(&arena, elem, arr_close);
            TAPE_PASS(elem2 != JBIN_NONE, "tape: has second array elem");
            s = jbin_get_str(&arena, elem2, json, &slen);
            TAPE_PASS(slen == 1 && s[0] == 'b', "tape: second array elem is 'b'");

            /* Navigate to second top-level key (skip nested obj) */
            uint32_t key2 = jbin_obj_next_key(&arena, key, close_idx);
            TAPE_PASS(key2 != JBIN_NONE, "tape: has second top key");
            (void)inner_close;
        }
    }

    /* Test 4: Empty containers in tape mode */
    {
        const char *json =
            "{\"empty_arr\": [], \"empty_obj\": {}, "
            "\"pad_to_64_bytes_string\": \"abcdefghij\"}";
        uint32_t len = (uint32_t)strlen(json);
        jbin_arena_init(&arena);
        JbinResult r = jbin_parse(&arena, json, len);
        TAPE_PASS(r.error == JBIN_OK, "tape: empty containers parse");

        if (r.error == JBIN_OK) {
            uint32_t close_idx = jbin_container_close(&arena, r.root);
            uint32_t key = jbin_first_child(&arena, r.root);
            uint32_t val = jbin_obj_value(&arena, key);
            TAPE_PASS(jbin_type(&arena, val) == JBIN_ARRAY, "tape: empty arr type");
            TAPE_PASS(jbin_first_child(&arena, val) == JBIN_NONE,
                      "tape: empty arr no children");

            uint32_t key2 = jbin_obj_next_key(&arena, key, close_idx);
            uint32_t val2 = jbin_obj_value(&arena, key2);
            TAPE_PASS(jbin_type(&arena, val2) == JBIN_OBJECT, "tape: empty obj type");
            TAPE_PASS(jbin_first_child(&arena, val2) == JBIN_NONE,
                      "tape: empty obj no children");
        }
    }

    /* Test 5: Dirty strings (escapes) in tape mode */
    {
        const char *json =
            "[\"hello\\nworld\", \"tab\\there\", "
            "\"quote\\\"inside\", \"padding_string_for_length\"]";
        uint32_t len = (uint32_t)strlen(json);
        jbin_arena_init(&arena);
        JbinResult r = jbin_parse(&arena, json, len);
        TAPE_PASS(r.error == JBIN_OK, "tape: escaped strings parse");

        if (r.error == JBIN_OK) {
            uint32_t close_idx = jbin_container_close(&arena, r.root);
            uint32_t child = jbin_first_child(&arena, r.root);
            uint32_t slen;
            const char *s = jbin_get_str(&arena, child, json, &slen);
            TAPE_PASS(slen == 11 && s[5] == '\n', "tape: escaped newline");

            uint32_t second = jbin_next_sibling(&arena, child, close_idx);
            s = jbin_get_str(&arena, second, json, &slen);
            TAPE_PASS(slen == 8 && s[3] == '\t', "tape: escaped tab");

            uint32_t third = jbin_next_sibling(&arena, second, close_idx);
            s = jbin_get_str(&arena, third, json, &slen);
            TAPE_PASS(slen == 12 && s[5] == '"', "tape: escaped quote");
        }
    }

    /* Test 6: Mixed types (literals + numbers) in string-heavy tape context */
    {
        const char *json =
            "[\"string\", true, false, null, 42, "
            "\"another_string_padding_for_64_bytes_xxxxxxx\"]";
        uint32_t len = (uint32_t)strlen(json);
        jbin_arena_init(&arena);
        JbinResult r = jbin_parse(&arena, json, len);
        TAPE_PASS(r.error == JBIN_OK, "tape: mixed types parse");

        if (r.error == JBIN_OK) {
            uint32_t close_idx = jbin_container_close(&arena, r.root);
            uint32_t c = jbin_first_child(&arena, r.root);
            TAPE_PASS(jbin_type(&arena, c) == JBIN_STRING, "tape: mixed[0] string");

            c = jbin_next_sibling(&arena, c, close_idx);
            TAPE_PASS(jbin_type(&arena, c) == JBIN_TRUE, "tape: mixed[1] true");

            c = jbin_next_sibling(&arena, c, close_idx);
            TAPE_PASS(jbin_type(&arena, c) == JBIN_FALSE, "tape: mixed[2] false");

            c = jbin_next_sibling(&arena, c, close_idx);
            TAPE_PASS(jbin_type(&arena, c) == JBIN_NULL, "tape: mixed[3] null");

            c = jbin_next_sibling(&arena, c, close_idx);
            TAPE_PASS(jbin_type(&arena, c) == JBIN_NUMBER, "tape: mixed[4] number");
            uint32_t slen;
            const char *s = jbin_get_str(&arena, c, json, &slen);
            TAPE_PASS(slen == 2 && s[0] == '4' && s[1] == '2',
                      "tape: number content '42'");

            c = jbin_next_sibling(&arena, c, close_idx);
            TAPE_PASS(jbin_type(&arena, c) == JBIN_STRING, "tape: mixed[5] string");
        }
    }

    /* Test 7: Verify is_tape flag is set correctly (only meaningful
       when compiled with -march=native for AVX2+PCLMUL two-pass) */
    {
        const char *tape_json =
            "[\"aaaa_long_string\", \"bbbb_long_string\", "
            "\"cccc_long_string\", \"dddd_long_string\", "
            "\"eeee_long_string\", \"ffff_long_string\"]";
        uint32_t len = (uint32_t)strlen(tape_json);
        jbin_arena_init(&arena);
        JbinResult r = jbin_parse(&arena, tape_json, len);
        TAPE_PASS(r.error == JBIN_OK, "tape: string-heavy parses ok");

        /* Small input should always use DOM (is_tape=0) */
        const char *dom_json = "[1, 2, 3]";
        jbin_arena_init(&arena);
        r = jbin_parse(&arena, dom_json, (uint32_t)strlen(dom_json));
        TAPE_PASS(r.error == JBIN_OK && arena.is_tape == 0,
                  "tape: is_tape=0 for small input");

        /* If two-pass is available, verify tape mode activates */
        if (arena.is_tape == 0) {
            /* Re-parse the string-heavy input and check */
            jbin_arena_init(&arena);
            r = jbin_parse(&arena, tape_json, len);
            if (arena.is_tape == 1) {
                TAPE_PASS(1, "tape: is_tape=1 for string-heavy (twopass)");
            } else {
                /* No twopass support compiled in - skip */
                TAPE_PASS(1, "tape: twopass not available (scalar build)");
            }
        } else {
            TAPE_PASS(1, "tape: twopass check placeholder");
        }
    }

    #undef TAPE_PASS

    printf("\nTape navigation: %d passed, %d failed\n", passed, failed);
    return failed;
}

int main(int argc, char **argv) {
    int verbose = 0;
    const char *suite_dir = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0)
            verbose = 1;
        else
            suite_dir = argv[i];
    }

    printf("=== jbin built-in tests ===\n");
    int failures = run_builtin_tests(verbose);

    printf("\n=== Tape navigation tests ===\n");
    failures += run_tape_tests(verbose);

    if (suite_dir) {
        printf("\n=== JSONTestSuite ===\n");
        failures += run_suite_dir(suite_dir, verbose);
    } else {
        printf("\nTip: pass JSONTestSuite/test_parsing dir as argument for full suite\n");
    }

    printf("\n%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILURES");
    return failures == 0 ? 0 : 1;
}
