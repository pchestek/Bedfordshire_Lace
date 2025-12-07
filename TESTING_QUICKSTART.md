# Testing Quick Start Guide

## Setup (5 minutes)

1. **Open test file in Inkscape**:
   ```bash
   inkscape test_shapes.svg
   ```

2. **Verify extension is available**:
   - Menu: Extensions > Textile Arts > Bedfordshire Lace
   - If not found, check file locations and restart Inkscape

## Quick Test Sequence (30 minutes)

### Test 1: Basic Tape (5 min)

**Steps**:
1. Select path `tape-curve` (red curved line)
2. Extensions > Textile Arts > Bedfordshire Lace
3. Settings: Element Type = **Tape**, Thread size = 60, Pairs = 8, Width = 10mm
4. Run

**Verify**:
- [ ] Two parallel edges following curve
- [ ] Gray pricking points alternating left/right
- [ ] Spacing ~1mm
- [ ] Blue dashed control path visible

**Test thread size variation**:
5. Select `tape-long` path
6. Run with Thread size = **40** → expect wider spacing (~1.5mm)
7. Select same path again
8. Run with Thread size = **120** → expect tighter spacing (~0.5mm)

### Test 2: Tally Rectangle (3 min)

**Steps**:
1. Select rectangle `tally-rect-simple`
2. Element Type = **Tally (Rectangular)**
3. Run

**Verify**:
- [ ] Rectangle outline matches original
- [ ] Gray dot at top center (entry)
- [ ] Gray dot at bottom center (exit)
- [ ] 4 red outline circles at corners

**Test rotated**:
4. Select `tally-rect-rotated`
5. Run with same settings
6. Verify rotation preserved

### Test 3: Tally Leaf (5 min)

**Steps**:
1. Select ellipse `tally-leaf-vertical`
2. Element Type = **Tally (Leaf-shaped)**
3. Run

**Verify**:
- [ ] Lens shape with pointed top/bottom
- [ ] Gray dot at top point
- [ ] Gray dot at bottom point
- [ ] 2 red markers at points

**Test horizontal**:
4. Select `tally-leaf-horizontal`
5. Run → should auto-rotate 90°

**Test rotated**:
6. Select `tally-leaf-rotated`
7. Run → should preserve rotation

**Test group**:
8. Select group `tally-leaf-group`
9. Run → should create 3 separate leaf tallies

### Test 4: Plait (3 min)

**Steps**:
1. Select path `plait-simple`
2. Element Type = **Plait**, Picots = **false**
3. Run

**Verify**:
- [ ] STRAIGHT line from start to end (ignores curve)
- [ ] Red circles at both ends
- [ ] No pricking along length

**Test picots**:
4. Select `plait-picot`
5. Element Type = **Plait**, Picots = **true**
6. Run

**Verify**:
- [ ] 2 hollow circles at midpoint
- [ ] 1 gray filled dot at center
- [ ] Circles perpendicular to line

### Test 5: Connections (10 min)

**Steps - Create elements first**:
1. Select `conn-tape-1`
2. Create Tape (thread 60, pairs 8, width 10mm)
3. Note positions of pricking points at end of tape

4. Select `conn-tally-1`
5. Create Tally (Rectangular)
6. Note position of entry point (top center)

**Steps - Create connecting plait**:
7. Select `conn-plait-1` OR draw new path:
   - Start point near tape end pricking (~3-4mm away)
   - End point near tally entry (~3-4mm away)
8. Create Plait (picots = false, **snap distance = 5mm**)

**Verify**:
- [ ] Plait endpoints snap to pricking points
- [ ] Tape pricking turns **GREEN**
- [ ] Tally entry turns **GREEN**
- [ ] Other pricking stay gray

**Test snap distance**:
9. Draw plait 7mm away from any pricking
10. Run with snap distance = 5mm → should NOT snap
11. Select same path
12. Run with snap distance = 10mm → SHOULD snap

### Test 6: Regeneration (4 min)

**Steps**:
1. Select `regen-test`
2. Create Tape with pairs = **8**
3. Note element ID in XML (optional)
4. **Select original path `regen-test` again**
5. Run with pairs = **16** (double width)

**Verify**:
- [ ] Old tape removed
- [ ] New tape created
- [ ] Width doubled
- [ ] No duplicate elements

**Test with connections**:
6. Create tape
7. Create plait connecting to it (pricking turns green)
8. Regenerate tape (select original path, run again)
9. Verify pricking still green (connection preserved via control_path_id)

## Validation (2 min)

**Run automated validator**:
```bash
python test_validator.py test_shapes.svg
```

**Expected output**:
- ✓ 'Lace Pattern' layer exists
- ✓ All metadata fields present
- ✓ Connection count matches green pricking count
- No errors

## Common Issues Checklist

### Extension doesn't appear in menu
- [ ] Files in correct directory: `~/.config/inkscape/extensions/`
- [ ] Both .py and .inx files present
- [ ] Restart Inkscape after installing

### "Please select a shape first"
- [ ] Did you select a path/shape before running?
- [ ] Selection visible in Inkscape?

### Tape/Plait requires path error
- [ ] Element type matches shape type
- [ ] For Tape: must be path (use Bezier tool or convert shape to path)
- [ ] For Plait: must be path

### Plait not snapping
- [ ] Are you within snap distance? (default 5mm)
- [ ] Is target pricking already active? (already has connection)
- [ ] Create tape/tally first, then plait

### Pricking not turning green
- [ ] Did plait snap? (endpoints should align with pricking exactly)
- [ ] Check metadata: plait should have start_connection/end_connection
- [ ] Try regenerating the tape/tally

### Transform issues (rotated shapes wrong)
- [ ] For rectangles: keep as native Rectangle (don't convert to path)
- [ ] For ellipses: works with rotated ellipses
- [ ] Check if transform attribute present on element group

## Advanced Testing

### Measure spacing
Use Inkscape's measurement tool (M key):
1. Measure distance between consecutive pricking points
2. Compare to formula: `spacing = 2 × (30 / thread_size)`
   - Thread 60: 1.0mm
   - Thread 40: 1.5mm
   - Thread 120: 0.5mm

### Inspect metadata
View > XML Editor (Shift+Ctrl+X):
1. Find lace element groups
2. Look for `data-lace-metadata` attribute
3. Verify JSON structure matches TESTING.md

### Check colors
Use dropper tool (F7):
- Gray pricking: #666666
- Green pricking: #00FF00
- Red markers: #FF0000
- Control path: #0000FF

## Test Results Template

Copy this for tracking:

```
BEDFORDSHIRE LACE EXTENSION - TEST RESULTS
Date: ___________
Tester: ___________

[ ] Test 1: Basic Tape - PASS / FAIL
    Notes: _________________________________

[ ] Test 2: Tally Rectangle - PASS / FAIL
    Notes: _________________________________

[ ] Test 3: Tally Leaf - PASS / FAIL
    Notes: _________________________________

[ ] Test 4: Plait - PASS / FAIL
    Notes: _________________________________

[ ] Test 5: Connections - PASS / FAIL
    Notes: _________________________________

[ ] Test 6: Regeneration - PASS / FAIL
    Notes: _________________________________

[ ] Validator: PASS / FAIL
    Errors: ____________
    Warnings: __________

ISSUES FOUND:
1. _______________________________________
2. _______________________________________
3. _______________________________________

OVERALL: PASS / FAIL
```

## Next Steps After Quick Test

If quick tests pass:
- [ ] Run full TESTING.md test suite
- [ ] Test edge cases (very short paths, extreme parameters)
- [ ] Test with real lace patterns
- [ ] Performance test with many elements

If issues found:
- [ ] Document with bug template (see TESTING.md)
- [ ] Check bedfordshire_lace.py at line numbers from DEVELOPMENT.md
- [ ] Report to developer

## Getting Help

- **Documentation**: Read DEVELOPMENT.md for implementation details
- **Test Plan**: TESTING.md has comprehensive test cases
- **Validation**: Use test_validator.py for automated checks
- **Code**: Check bedfordshire_lace.py for element creation functions
