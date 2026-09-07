// spvtool: SPIRV-Tools 정적라이브러리로 어셈블/디스어셈블
// usage: spvtool asm <in.spvasm> <out.spv> | spvtool dis <in.spv> <out.spvasm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spirv-tools/libspirv.h"

int main(int argc, char **argv) {
    if (argc != 4 && !(argc == 3 && !strcmp(argv[1], "val"))) {
        fprintf(stderr, "usage: %s asm|dis in out | %s val in\n", argv[0], argv[0]); return 2; }
    const char *outpath = (argc == 4) ? argv[3] : NULL;
    FILE *f = fopen(argv[2], "rb");
    if (!f) { perror("open"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1); fread(buf, 1, n, f); buf[n] = 0; fclose(f);
    spv_context ctx = spvContextCreate(SPV_ENV_VULKAN_1_3);
    spv_binary bin = NULL; spv_text txt = NULL;
    spv_diagnostic diag = NULL;
    spv_result_t r;
    if (!strcmp(argv[1], "val")) {
        r = spvValidateBinary(ctx, (const uint32_t*)buf, (uint32_t)(n/4), &diag);
        if (r) { if (diag) spvDiagnosticPrint(diag); return 1; }
        printf("VALID\n"); return 0;
    }
    if (!strcmp(argv[1], "asm")) {
        r = spvTextToBinary(ctx, buf, n, &bin, &diag);
        if (r) { if (diag) spvDiagnosticPrint(diag); return 1; }
        FILE *o = fopen(outpath, "wb");
        fwrite(bin->code, 4, bin->wordCount, o); fclose(o);
    } else {
        r = spvBinaryToText(ctx, (const uint32_t*)buf, n/4,
                            SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | SPV_BINARY_TO_TEXT_OPTION_INDENT,
                            &txt, &diag);
        if (r) { if (diag) spvDiagnosticPrint(diag); return 1; }
        FILE *o = fopen(outpath, "wb"); fwrite(txt->str, 1, txt->length, o); fclose(o);
    }
    return 0;
}
