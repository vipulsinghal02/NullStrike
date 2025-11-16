# Documentation Streamlining Plan

## Current State Analysis

**Total Documentation: ~13,760 lines across 19 files**

### Size Breakdown by Directory:
- **dev/** - 5,342 lines (6 files) - 39% of total ⚠️ LARGEST
- **advanced/** - 2,889 lines (3 files) - 21% of total ⚠️
- **theory/** - 1,294 lines (3 files) - 9%
- **guide/** - 1,699 lines (3 files) - 12%
- **root/** - 695 lines (3 files) - 5%
- **api/** - 567 lines (1 file) - 4%
- **getting-started/** - 459 lines (1 file) - 3%
- **results/** - 430 lines (1 file) - 3%
- **examples/** - 385 lines (1 file) - 3%

## Problem Areas

### 🔴 Bloated Sections (60% of total docs):
1. **dev/** (5,342 lines) - Developer documentation is too detailed
   - api-development.md (1,033 lines)
   - testing.md (1,012 lines)
   - performance.md (902 lines)
   - release.md (947 lines)
   - contributing.md (642 lines)
   - architecture.md (806 lines)

2. **advanced/** (2,889 lines) - Advanced topics are verbose
   - troubleshooting.md (1,013 lines)
   - batch.md (982 lines)
   - workflows.md (894 lines)

## Proposed Streamlined Structure

### Target: Reduce to ~6,000-7,000 lines (50% reduction)

```
docs/
├── index.md                    (Keep ~150 lines) - Landing page
├── quickstart.md               (Keep ~200 lines) - 5-min start
├── installation.md             (Keep ~200 lines) - Setup
│
├── guide/                      (Reduce to ~1,200 lines)
│   ├── basics.md              (NEW: 400 lines) - Core concepts + CLI
│   ├── models.md              (Trim to 400 lines) - Model definition
│   └── advanced.md            (NEW: 400 lines) - Batch + workflows
│
├── theory/                     (Keep ~1,300 lines)
│   ├── overview.md            (Keep)
│   ├── nullspace.md           (Keep)
│   └── strike-goldd.md        (Keep)
│
├── examples/                   (Expand to ~800 lines)
│   ├── quick-start.md         (NEW: 300 lines)
│   ├── c2m-walkthrough.md     (NEW: 300 lines)
│   └── custom-model.md        (NEW: 200 lines)
│
├── reference/                  (NEW: ~1,500 lines)
│   ├── cli.md                 (NEW: 400 lines) - CLI reference
│   ├── api.md                 (Condensed: 500 lines) - API docs
│   ├── configuration.md       (NEW: 300 lines) - All config options
│   └── troubleshooting.md     (Trim to 300 lines) - Common issues only
│
└── contributing.md             (Consolidate to ~800 lines)
    └── All dev docs merged here

REMOVE:
├── ❌ dev/ (entire directory - move to CONTRIBUTING.md)
├── ❌ advanced/ (merge into guide/advanced.md)
├── ❌ getting-started/ (merge into quickstart.md)
├── ❌ results/ (merge into examples)
```

## Consolidation Strategy

### Phase 1: Merge & Delete (Immediate ~50% reduction)

**Consolidate dev/ → contributing.md**
- Keep: Essential contributing guidelines (200 lines)
- Keep: Architecture overview diagram (100 lines)
- Keep: Testing basics (150 lines)
- Move to GitHub Wiki: Detailed API development, performance tuning, release process
- **DELETE**: api-development.md, performance.md, release.md
- **TRIM**: testing.md to essentials only

**Consolidate advanced/ → guide/advanced.md**
- Merge workflows.md + batch.md → guide/advanced.md (400 lines)
- **TRIM**: troubleshooting.md → reference/troubleshooting.md (top 20 issues only)

**Consolidate getting-started/ → quickstart.md**
- Merge first-analysis.md content into quickstart.md
- **DELETE**: getting-started/ directory

### Phase 2: Restructure (Better organization)

**Create guide/basics.md** (NEW)
- Core concepts (100 lines)
- CLI usage (150 lines)
- Configuration basics (150 lines)

**Expand examples/** (User-focused)
- Move results/interpretation.md → examples/
- Add step-by-step walkthroughs
- Add troubleshooting specific to examples

**Create reference/** (Quick lookup)
- All CLI commands
- All API functions
- All config options
- FAQs only

### Phase 3: Content Trimming Guidelines

For each remaining file, apply these rules:

1. **Remove redundancy**: If it's in 2+ places, keep it in 1
2. **Cut verbosity**: Replace long explanations with concise bullets
3. **Move to code**: Put detailed API docs in docstrings, not markdown
4. **Link externally**: Reference GitHub Issues/Wiki for niche topics
5. **Keep examples**: Users prefer examples over explanations

## New Documentation Map

```
📚 NullStrike Documentation (~6,500 lines total)

🏠 HOME
├─ index.md (150 lines)
└─ quickstart.md (200 lines)

📖 USER GUIDE (~2,400 lines)
├─ installation.md (200 lines)
├─ guide/basics.md (400 lines)
├─ guide/models.md (400 lines)
├─ guide/advanced.md (400 lines)
├─ examples/quick-start.md (300 lines)
├─ examples/c2m-walkthrough.md (300 lines)
└─ examples/custom-model.md (200 lines)

🧮 THEORY (~1,300 lines)
├─ theory/overview.md
├─ theory/nullspace.md
└─ theory/strike-goldd.md

📋 REFERENCE (~2,100 lines)
├─ reference/cli.md (400 lines)
├─ reference/api.md (500 lines)
├─ reference/configuration.md (300 lines)
└─ reference/troubleshooting.md (300 lines)
└─ contributing.md (600 lines)

🗑️ ARCHIVED (Move to GitHub Wiki)
├─ Detailed API development
├─ Performance tuning guide
├─ Release process
└─ Extensive troubleshooting database
```

## Implementation Checklist

### Step 1: Backup
- [ ] Create `docs_backup/` with full copy
- [ ] Commit current state before changes

### Step 2: Delete/Move
- [ ] Delete `dev/` (6 files)
- [ ] Delete `advanced/` (3 files)
- [ ] Delete `getting-started/` (1 file)
- [ ] Delete `results/` (1 file)
- [ ] Delete `api/` (1 file)

### Step 3: Create New Structure
- [ ] Create `guide/basics.md`
- [ ] Create `guide/advanced.md`
- [ ] Create `reference/` directory
- [ ] Create `reference/cli.md`
- [ ] Create `reference/api.md`
- [ ] Create `reference/configuration.md`
- [ ] Create `reference/troubleshooting.md`
- [ ] Expand `examples/` with new files

### Step 4: Consolidate Content
- [ ] Merge dev/* → contributing.md
- [ ] Merge advanced/* → guide/advanced.md + reference/troubleshooting.md
- [ ] Trim guide/models.md (630 → 400 lines)
- [ ] Update index.md with new structure

### Step 5: Update Navigation
- [ ] Update mkdocs.yml nav structure
- [ ] Update README.md links
- [ ] Update cross-references between docs

### Step 6: Quality Check
- [ ] Build docs locally (`mkdocs serve`)
- [ ] Check all internal links work
- [ ] Verify examples are clear
- [ ] Test that quickstart works end-to-end

## Expected Outcomes

**Before:**
- 13,760 lines across 19 files in 9 directories
- Fragmented information
- Hard to navigate
- Too much detail for users
- Developer docs dominate

**After:**
- ~6,500 lines across 14 files in 4 directories
- Clear user/reference/theory separation
- Quick navigation
- Example-driven learning
- Developer docs condensed

**Reduction: 53% fewer lines, 26% fewer files, cleaner structure**

## Alternative: Minimal Documentation

If you want even more aggressive trimming:

```
docs/
├── index.md (200 lines) - Everything a user needs to know
├── quickstart.md (200 lines)
├── theory.md (500 lines) - All theory in one place
├── examples.md (400 lines) - All examples in one place
├── reference.md (400 lines) - CLI + API + Config
└── contributing.md (300 lines)

Total: 6 files, ~2,000 lines (85% reduction!)
```

This "single page per topic" approach is very maintainable but loses some organization.

## Recommendation

**Go with the "Streamlined Structure" plan** (6,500 lines, 50% reduction):
- Maintains good organization
- Easier to navigate than single-page
- Removes bloat without losing essential info
- Better SEO (more pages)
- Room to grow if needed

Next steps: Review this plan, then I'll help execute it!
