# Bedfordshire Lace Extension - Test Plan

## Overview
This document provides comprehensive testing procedures for all implemented lace elements. Each test includes setup instructions, expected results, and verification criteria.

---

## Test Environment Setup

### Prerequisites
1. Inkscape installed with extension properly configured
2. Test SVG file with various control shapes prepared
3. Access to Extensions > Textile Arts > Bedfordshire Lace

### Creating Test Control Shapes
Before testing, create an Inkscape document with:
- **Paths**: Use Bezier tool to create straight, curved, and S-curved paths
- **Rectangles**: Create native rectangles and rotated rectangles
- **Ellipses**: Create vertical ellipses (ry > rx), horizontal ellipses (rx > ry), and rotated ellipses
- **Groups**: Create groups containing multiple shapes

---

## 1. Tape Element Tests

### Test 1.1: Basic Tape Generation
**Objective**: Verify tape creates properly with default settings

**Setup**:
1. Draw a simple curved path (Bezier tool)
2. Select the path
3. Open extension: Extensions > Textile Arts > Bedfordshire Lace
4. Settings: Element Type = Tape, Thread size = 60, Tape pairs = 8, Base width = 10mm

**Expected Results**:
- Two parallel edge lines following the path curvature
- Gray pricking points alternating between left and right edges
- Points should be evenly spaced along centerline
- Spacing = 2 × (30 / 60) = 1mm between points
- Blue dashed control path visible at 50% opacity
- Width approximately 10mm (measure between edges)

**Verification**:
- [ ] Edge lines are smooth and parallel
- [ ] Pricking points alternate left-right-left-right
- [ ] Spacing is consistent (~1mm)
- [ ] Width matches setting (~10mm)
- [ ] Control path is visible and blue

### Test 1.2: Thread Size Variation
**Objective**: Verify spacing changes with thread size

**Setup**: Same path, test with different thread sizes

**Test Cases**:
| Thread Size | Expected Spacing (mm) | Expected Appearance |
|-------------|----------------------|---------------------|
| 40 (coarse) | 2 × (30/40) = 1.5mm | Wider spacing |
| 60 (medium) | 2 × (30/60) = 1.0mm | Medium spacing |
| 80 (fine)   | 2 × (30/80) = 0.75mm | Tighter spacing |
| 120 (v.fine)| 2 × (30/120) = 0.5mm | Very tight spacing |

**Verification**:
- [ ] Coarser thread → wider spacing
- [ ] Finer thread → tighter spacing
- [ ] Spacing formula works correctly
- [ ] Points remain on alternating edges

### Test 1.3: Tape Width Scaling
**Objective**: Verify width changes with pair count and width setting

**Test Cases**:
1. **Varying pair count** (width = 10mm):
   - 2 pairs: narrow tape
   - 8 pairs: medium tape
   - 20 pairs: wide tape
   - Width should scale proportionally: `actual_width = base_width × (pairs / initial_pairs)`

2. **Varying base width** (8 pairs):
   - 5mm: half width
   - 10mm: default width
   - 20mm: double width

**Verification**:
- [ ] More pairs → wider tape
- [ ] Width setting controls actual width
- [ ] Edge lines remain parallel at all widths
- [ ] Pricking density doesn't change with width

### Test 1.4: Path Complexity
**Objective**: Verify tape handles various path shapes

**Test Cases**:
1. Straight line
2. Simple curve
3. S-curve
4. Sharp turn
5. Loop/spiral

**Verification**:
- [ ] Edge lines follow path correctly
- [ ] No edge crossings or inversions
- [ ] Points maintain regular spacing
- [ ] Alternating pattern maintained throughout
- [ ] Sharp curves handled gracefully (inner curve tighter, outer wider)

### Test 1.5: Control Path Visibility
**Objective**: Verify show_control_path setting works

**Test Cases**:
1. show_control_path = true: Blue dashed line at 50% opacity
2. show_control_path = false: Control path hidden (display=none)

**Verification**:
- [ ] Setting true shows blue dashed control path
- [ ] Setting false hides control path completely
- [ ] Lace pattern visible in both cases

---

## 2. Tally Rectangle Tests

### Test 2.1: Basic Rectangle Tally
**Objective**: Verify rectangular tally creation

**Setup**:
1. Draw a rectangle (Rectangle tool)
2. Select rectangle
3. Element Type = Tally (Rectangular)

**Expected Results**:
- Rectangle outline (black, 0.5pt stroke)
- Gray pricking point at top center (entry)
- Gray pricking point at bottom center (exit)
- Red outline circles at all 4 corners (connection markers)
- Pricking points: 0.8mm radius
- Connection markers: 1.5mm radius

**Verification**:
- [ ] Rectangle matches original dimensions
- [ ] Entry point at top center
- [ ] Exit point at bottom center
- [ ] 4 corner markers present
- [ ] All markers are red outlines (not filled)

### Test 2.2: Rotated Rectangle
**Objective**: Verify transform preservation for native rectangles

**Setup**:
1. Draw rectangle
2. Rotate it 45° using transform tool (NOT converted to path)
3. Apply tally_rect

**Expected Results**:
- Tally group has transform attribute matching rotation
- Rectangle rendered in local coordinates
- Entry/exit points in metadata are world coordinates (transformed)
- Visual appearance matches rotated original

**Verification**:
- [ ] Tally appears rotated correctly
- [ ] Transform attribute present on tally group
- [ ] Entry/exit points in correct world positions
- [ ] Pricking points and markers align properly

### Test 2.3: Path-based Rectangle
**Objective**: Verify tally works with converted rectangles

**Setup**:
1. Draw rectangle
2. Convert to path (Path > Object to Path)
3. Apply tally_rect

**Expected Results**:
- Uses bounding box approach
- No transform on tally group
- Rectangle aligned to bounding box
- All coordinates in world space

**Verification**:
- [ ] Tally created successfully
- [ ] Dimensions match bounding box
- [ ] Entry/exit at correct positions
- [ ] Works for rotated paths

### Test 2.4: Arbitrary Shapes
**Objective**: Verify tally_rect works with non-rectangle shapes

**Setup**:
1. Draw circle, convert to path
2. Apply tally_rect

**Expected Results**:
- Rectangle fitted to bounding box of shape
- Entry at top center of bbox
- Exit at bottom center of bbox

**Verification**:
- [ ] Bounding box rectangle created
- [ ] Entry/exit positioned correctly
- [ ] Markers at bbox corners

---

## 3. Tally Leaf Tests

### Test 3.1: Basic Leaf Tally (Vertical Ellipse)
**Objective**: Verify lens shape creation from vertical ellipse

**Setup**:
1. Draw ellipse with ry > rx (taller than wide), e.g., rx=20, ry=40
2. Element Type = Tally (Leaf-shaped)

**Expected Results**:
- Vesica piscis (lens) outline
- Pointed ends at top and bottom
- Gray pricking at top point (entry)
- Gray pricking at bottom point (exit)
- Red outline markers at top and bottom
- Aspect ratio warning if ry/rx < 1.5

**Verification**:
- [ ] Lens shape with pointed ends
- [ ] Smooth curved sides
- [ ] Entry at top point
- [ ] Exit at bottom point
- [ ] 2 markers (not 4 like rectangle)

### Test 3.2: Horizontal Ellipse
**Objective**: Verify automatic rotation for horizontal ellipses

**Setup**:
1. Draw ellipse with rx > ry (wider than tall), e.g., rx=40, ry=20
2. Apply tally_leaf

**Expected Results**:
- Extension detects rx > ry
- Automatically rotates 90° to make lens vertical
- Lens points left and right in final rendering
- Transform includes rotation

**Verification**:
- [ ] Lens oriented horizontally (points left-right)
- [ ] Dimensions swapped correctly
- [ ] Entry/exit at horizontal points
- [ ] Transform includes 90° rotation

### Test 3.3: Rotated Ellipse
**Objective**: Verify transform handling for rotated ellipses

**Setup**:
1. Draw vertical ellipse (ry > rx)
2. Rotate 30° using transform tool
3. Apply tally_leaf

**Expected Results**:
- Lens follows ellipse rotation
- Entry/exit points rotated correctly
- Transform composition correct
- Metadata has world coordinates

**Verification**:
- [ ] Lens rotated to match ellipse
- [ ] Entry/exit in correct world positions
- [ ] Visual alignment perfect
- [ ] Transform attribute composed correctly

### Test 3.4: Grouped Ellipses
**Objective**: Verify group recursion processes individual ellipses

**Setup**:
1. Draw 3 ellipses
2. Group them (Ctrl+G)
3. Select group
4. Apply tally_leaf

**Expected Results**:
- Extension processes each ellipse separately
- Creates 3 separate tally_leaf elements
- Each has independent metadata
- Parent group transforms included

**Verification**:
- [ ] 3 tally elements created
- [ ] Each ellipse processed individually
- [ ] Group transform applied correctly
- [ ] All ellipses have lace patterns

### Test 3.5: Aspect Ratio Warning
**Objective**: Verify warning for near-circular ellipses

**Setup**:
1. Draw ellipse with rx≈ry (aspect ratio < 1.5), e.g., rx=30, ry=35
2. Apply tally_leaf

**Expected Results**:
- Warning message: "Leaf tally works best when one dimension is at least 1.5× the other. Current ratio: X.XX"
- Tally still created (warning, not error)
- Lens shape may look bulbous

**Verification**:
- [ ] Warning displayed
- [ ] Aspect ratio calculated correctly
- [ ] Tally still generated
- [ ] Message includes actual ratio

### Test 3.6: Non-Ellipse Shapes
**Objective**: Verify fallback to bounding box for non-ellipse shapes

**Setup**:
1. Draw rectangle or convert circle to path
2. Apply tally_leaf

**Expected Results**:
- Uses bounding box approach
- Lens fitted to bbox dimensions
- No transform on tally group

**Verification**:
- [ ] Lens created from bbox
- [ ] Entry/exit at top/bottom of bbox
- [ ] Works but may not handle rotations optimally

---

## 4. Plait Tests

### Test 4.1: Basic Plait (No Snapping)
**Objective**: Verify standalone plait creation

**Setup**:
1. Draw a path (straight or curved)
2. Element Type = Plait, Picots = false

**Expected Results**:
- STRAIGHT line from start to end (ignores path curvature)
- Black line, 0.5pt stroke
- Red outline circles at start and end (1.5mm radius)
- No pricking along length
- Blue dashed control path

**Verification**:
- [ ] Line is perfectly straight (not curved)
- [ ] Start/end markers present
- [ ] No intermediate pricking points
- [ ] Line drawn start to end regardless of path shape

### Test 4.2: Plait with Picots
**Objective**: Verify picot decoration

**Setup**:
1. Draw path
2. Element Type = Plait, Picots = true

**Expected Results**:
- Straight line with picots at midpoint
- 2 hollow circles (0.6mm radius) offset perpendicular to line
- 1 filled gray dot (0.4mm radius) at center
- Picot offset: 1.5mm from centerline

**Verification**:
- [ ] Picots at exact midpoint
- [ ] Left and right circles hollow (outline only)
- [ ] Center pin filled gray
- [ ] Circles perpendicular to line direction
- [ ] Symmetric placement

### Test 4.3: Plait Snapping to Tape
**Objective**: Verify auto-snapping to pricking points

**Setup**:
1. Create tape element first
2. Draw path with start/end near tape pricking points (within 5mm)
3. Create plait

**Expected Results**:
- Plait endpoints snap to nearest pricking points
- Tape pricking changes from gray to green
- Metadata records connection:
  - start_connection: {element_id, control_path_id, pricking_index, ...}
  - end_connection: {...}

**Verification**:
- [ ] Endpoints align with tape pricking
- [ ] Connected pricking turns green
- [ ] Other pricking points stay gray
- [ ] Metadata has connection info
- [ ] control_path_id used (not element_id)

### Test 4.4: Snap Distance Testing
**Objective**: Verify snap_distance parameter works

**Test Cases**:
1. **Within snap distance** (snap_distance = 5mm):
   - Draw plait endpoint 3mm from pricking → should snap

2. **Outside snap distance**:
   - Draw plait endpoint 7mm from pricking → should NOT snap

3. **Varying snap distance**:
   - Test with snap_distance = 2mm, 10mm, 20mm

**Verification**:
- [ ] Snaps when distance < snap_distance
- [ ] Doesn't snap when distance > snap_distance
- [ ] Distance threshold works correctly
- [ ] Changing setting changes behavior

### Test 4.5: Multiple Plaits to Same Pricking
**Objective**: Verify only one plait can connect per pricking (is_active check)

**Setup**:
1. Create tape
2. Create plait connecting to pricking point
3. Try to create second plait to same point

**Expected Results**:
- First plait snaps, marks point as active (green)
- Second plait cannot snap to same point (already active)
- Second plait endpoints remain at drawn position

**Verification**:
- [ ] First plait connects successfully
- [ ] Point turns green
- [ ] Second plait doesn't snap to same point
- [ ] is_active flag prevents duplicate connections

### Test 4.6: Plait Between Two Tapes
**Objective**: Verify plait can connect both ends

**Setup**:
1. Create two tape elements
2. Draw plait path with start near first tape, end near second tape
3. Create plait

**Expected Results**:
- Both endpoints snap to respective tapes
- Both pricking points turn green
- start_connection and end_connection both populated
- Straight line connects the two points

**Verification**:
- [ ] Both ends snap correctly
- [ ] Two green pricking points
- [ ] Metadata has both connections
- [ ] Visual alignment perfect

---

## 5. Connection Detection Tests

### Test 5.1: Tape Pricking Color Update
**Objective**: Verify pricking points change color when connected

**Setup**:
1. Create tape (all pricking gray initially)
2. Create plait connecting to one pricking
3. Verify color change

**Expected Results**:
- Connected pricking: green (#00FF00)
- Other pricking: gray (#666666)
- Color updates on plait creation

**Verification**:
- [ ] Single green point where plait connects
- [ ] All other points gray
- [ ] Color clearly distinguishable

### Test 5.2: Tally Entry/Exit Connection
**Objective**: Verify tally pricking points respond to connections

**Setup**:
1. Create tally (rect or leaf)
2. Create plait to entry point
3. Create plait to exit point

**Expected Results**:
- Entry point turns green when plait connects (pricking_index = 0)
- Exit point turns green when plait connects (pricking_index = 1)

**Verification**:
- [ ] Entry connection detected
- [ ] Exit connection detected
- [ ] Colors update correctly
- [ ] Metadata tracking works

### Test 5.3: Connection Across Regeneration
**Objective**: Verify connections persist when regenerating elements

**Setup**:
1. Create tape
2. Create plait connecting to tape
3. Verify connection (green pricking)
4. Re-run extension on original tape control path (regenerate)
5. Check connection preserved

**Expected Results**:
- Old tape element deleted
- New tape element created
- Connection detected via control_path_id
- Pricking still green after regeneration

**Verification**:
- [ ] Old element removed
- [ ] New element created
- [ ] Green pricking point restored
- [ ] control_path_id enables stable tracking

---

## 6. Regeneration Tests

### Test 6.1: Basic Regeneration
**Objective**: Verify elements can be regenerated without duplication

**Setup**:
1. Create any lace element (tape/tally/plait)
2. Note the element_id
3. Re-select original control path
4. Run extension again with same settings

**Expected Results**:
- find_lace_element_by_control_path() finds old element
- Old element deleted
- New element created with new element_id
- control_path_id remains same

**Verification**:
- [ ] No duplicate elements
- [ ] Old element removed before creating new
- [ ] New element has fresh ID
- [ ] Metadata control_path_id unchanged

### Test 6.2: Regeneration with Setting Changes
**Objective**: Verify regeneration updates when settings change

**Test Cases**:
1. **Tape**: Create with 8 pairs, regenerate with 12 pairs
2. **Plait**: Create without picots, regenerate with picots
3. **Any**: Create with thread size 60, regenerate with size 40

**Expected Results**:
- New element reflects new settings
- Old element completely removed
- Parameters updated in metadata

**Verification**:
- [ ] Settings changes applied
- [ ] Visual appearance updated
- [ ] Metadata reflects new values
- [ ] Clean replacement (no artifacts)

### Test 6.3: Regeneration Preserves Connections
**Objective**: Verify connections work across regeneration (uses control_path_id)

**Setup**:
1. Create tape (tape1)
2. Create plait connecting to tape1
3. Regenerate tape1 (new element_id)
4. Check plait still shows as connected

**Expected Results**:
- Tape regenerated successfully
- Plait's connection still valid (uses control_path_id, not element_id)
- Connected pricking still green
- find_connections_to_control_path() works

**Verification**:
- [ ] Tape element_id changes
- [ ] Tape control_path_id unchanged
- [ ] Plait connection info still valid
- [ ] Green highlighting maintained

---

## 7. Multiple Element Interaction Tests

### Test 7.1: Mixed Elements in One Document
**Objective**: Verify multiple element types coexist

**Setup**:
Create in single document:
1. Tape (curved path)
2. Tally rectangle
3. Tally leaf
4. Plait
5. Leaf

**Expected Results**:
- All elements render correctly
- No interference between elements
- Each has proper metadata
- Lace Pattern layer contains all

**Verification**:
- [ ] All 5 element types present
- [ ] Each element independent
- [ ] No visual artifacts
- [ ] Metadata unique per element

### Test 7.2: Complex Lace Pattern
**Objective**: Build realistic pattern with connections

**Setup**:
1. Create tape with curve
2. Create 2 tally rectangles
3. Create 3 plaits connecting tape to tallies
4. Verify connections

**Expected Results**:
- Plaits snap to tape and tallies
- Multiple green pricking on tape
- Tally entry/exit points green when connected
- All metadata correct

**Verification**:
- [ ] All connections successful
- [ ] Green highlighting shows connections
- [ ] Pattern visually coherent
- [ ] Metadata tracks all relationships

### Test 7.3: Group Processing
**Objective**: Verify grouped shapes process individually

**Setup**:
1. Draw 5 ellipses
2. Group them
3. Select group
4. Apply tally_leaf

**Expected Results**:
- get_elements_recursively() expands group
- 5 separate tally_leaf elements created
- Each ellipse processed independently

**Verification**:
- [ ] All 5 ellipses processed
- [ ] Group recursion works
- [ ] 5 independent lace elements
- [ ] No errors from grouped selection

---

## 8. Edge Cases and Error Handling

### Test 8.1: Invalid Selections
**Objective**: Verify proper error messages

**Test Cases**:
1. **No selection**: Error "Please select a shape first"
2. **Wrong shape for tape**: Select circle, apply tape → Error "Tape requires a path object"
3. **Wrong shape for plait**: Select rectangle, apply plait → Error "Plait requires a path object"

**Verification**:
- [ ] Appropriate error messages
- [ ] Extension doesn't crash
- [ ] User guided to correct usage

### Test 8.2: Degenerate Paths
**Objective**: Handle edge cases gracefully

**Test Cases**:
1. **Zero-length path**: Single point → Error "Path is too short"
2. **Very short path**: 0.1mm path → May create minimal elements
3. **Self-intersecting path**: Figure-8 → Should handle without errors

**Verification**:
- [ ] Errors for invalid input
- [ ] No crashes on edge cases
- [ ] Graceful degradation where possible

### Test 8.3: Extreme Parameter Values
**Objective**: Test boundary conditions

**Test Cases**:
1. **Thread size**: Min (20), Max (200)
2. **Tape pairs**: Min (2), Max (20)
3. **Tape width**: Min (1mm), Max (50mm)
4. **Snap distance**: Min (1mm), Max (20mm)

**Verification**:
- [ ] All values within range work
- [ ] No division by zero
- [ ] No negative dimensions
- [ ] Reasonable visual output

---

## 9. Visual Quality Tests

### Test 9.1: Pricking Point Visibility
**Objective**: Verify pricking points are appropriately sized

**Expected**:
- Radius: 0.8mm
- Clearly visible but not overwhelming
- Gray (#666666) or green (#00FF00)
- Solid fill, no stroke

**Verification**:
- [ ] Points visible at 100% zoom
- [ ] Not too large (don't obscure pattern)
- [ ] Color distinguishable
- [ ] Consistent size across elements

### Test 9.2: Line Weights and Styles
**Objective**: Verify consistent styling

**Expected**:
- Element outlines: 0.5pt stroke, black
- Control paths: blue (#0000FF), dashed (5,5), 50% opacity
- Connection markers: red (#FF0000), 0.5pt stroke, 1.5mm radius

**Verification**:
- [ ] Line weights consistent
- [ ] Colors match specification
- [ ] Dashing pattern clear
- [ ] Professional appearance

### Test 9.3: Transform Accuracy
**Objective**: Verify transformed elements align perfectly

**Setup**:
1. Create rotated ellipse
2. Apply tally_leaf
3. Overlay original and generated
4. Check alignment

**Verification**:
- [ ] Perfect alignment with original
- [ ] No offset or rotation errors
- [ ] Entry/exit at exact points
- [ ] Transform math correct

---

## 10. Metadata Validation Tests

### Test 10.1: Metadata Structure
**Objective**: Verify metadata format for each element type

**Procedure**:
1. Create one of each element type
2. Use XML editor to examine data-lace-metadata attribute
3. Parse JSON and verify structure

**Expected Fields**:

**All elements**:
- element_type
- control_path_id
- pairs

**Tape**:
- initial_pairs, current_pairs, base_width_mm, splits[], pricking_ports[]

**Tally (both)**:
- width, height, entry_point[], exit_point[], connection_points[]

**Plait**:
- has_picots, start_point[], end_point[], connection_points[], start_connection, end_connection

**Leaf**:
- pricking_points[], entry_exit_point[]

**Verification**:
- [ ] All required fields present
- [ ] Valid JSON
- [ ] Data types correct
- [ ] Coordinates are [x, y] arrays

### Test 10.2: Connection Metadata
**Objective**: Verify connection info structure

**Setup**:
1. Create tape and plait connection
2. Examine plait metadata

**Expected in start_connection / end_connection**:
```json
{
  "point": [x, y],
  "element_id": "tape_123",
  "control_path_id": "path456",
  "element_type": "tape",
  "pricking_index": 5,
  "is_active": false
}
```

**Verification**:
- [ ] Connection object present
- [ ] All fields populated
- [ ] control_path_id for stability
- [ ] pricking_index correct

---

## Test Execution Checklist

### Before Testing
- [ ] Inkscape installed and running
- [ ] Extension files in correct location
- [ ] Test document prepared with control shapes
- [ ] DEVELOPMENT.md reviewed

### During Testing
- [ ] Document each test result
- [ ] Screenshot issues
- [ ] Note unexpected behaviors
- [ ] Record actual vs expected

### After Testing
- [ ] Summarize findings
- [ ] Create bug list if issues found
- [ ] Update DEVELOPMENT.md with test status
- [ ] Prioritize fixes

---

## Bug Report Template

When issues are found, use this template:

**Bug ID**: [Sequential number]
**Test**: [Test number and name]
**Severity**: [Critical / Major / Minor / Cosmetic]
**Description**: [What went wrong]
**Steps to Reproduce**:
1.
2.
3.

**Expected**: [What should happen]
**Actual**: [What actually happened]
**Screenshots**: [If applicable]
**Code Location**: [File:line if known]
**Suggested Fix**: [If obvious]

---

## Performance Considerations

### Test 11.1: Large Documents
**Objective**: Verify performance with many elements

**Setup**:
1. Create 50+ lace elements in one document
2. Add new plait (tests find_all_pricking_points)
3. Regenerate element (tests find_lace_element_by_control_path)

**Verification**:
- [ ] Acceptable performance (<5 seconds)
- [ ] No exponential slowdown
- [ ] All features still work

### Test 11.2: Complex Paths
**Objective**: Test path sampling performance

**Setup**:
1. Create very long curved path (100+ nodes)
2. Apply tape with fine thread (size 120)

**Verification**:
- [ ] Completes in reasonable time
- [ ] Path sampled correctly
- [ ] Spacing maintained throughout

---

## Conclusion

This test plan covers:
- ✅ All 5 element types (tape, tally_rect, tally_leaf, plait, leaf)
- ✅ All parameters and settings
- ✅ Connection system
- ✅ Regeneration
- ✅ Transform handling
- ✅ Group processing
- ✅ Edge cases
- ✅ Visual quality
- ✅ Metadata structure

Execute tests in order, documenting results for each. After completion, you'll have comprehensive validation of the extension's functionality and a clear list of any issues requiring fixes.
