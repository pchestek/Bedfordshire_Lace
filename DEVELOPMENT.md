# Bedfordshire Lace Extension Development

## Project Status

This Inkscape extension generates pricking patterns (pin placement diagrams) for Bedfordshire bobbin lace. Users draw control paths/shapes in Inkscape, and the extension converts them into authentic lace element patterns with proper pricking point placement.

## Current Implementation Status

### ✅ Completed Elements

#### 1. Tape
- **Thread pairs**: Variable (2-20, default 8)
- **Control shape**: Path (Bezier curves, open or closed)
- **Features**:
  - Alternating pricking points along edges (zigzag pattern)
  - Dynamic width based on pair count
  - Edge handling: Two separate closed paths for closed shapes
  - Pricking ports array in metadata
  - Connection tracking (turns green when plait connects)
  - Regeneration support (stable control_path_id)
  - Vertex detection in path sampling
  - Miter offset edge calculation to handle self-intersecting corners
- **Location**: `create_tape()` (lines ~715-1000+)
- **🚧 WORK IN PROGRESS - Corner Handling**:

  **Current Status (as of 2025-12-07)**:
  - Closed path support implemented
  - Vertex detection working (5 vertices detected for house-shape test)
  - Edge mitering implemented to avoid self-intersections
  - **CRITICAL ISSUE**: Pricking point positioning at corners

  **The Problem**:
  Traditional bobbin lace requires pricking points to be positioned at the **geometric angle bisector** of each corner:
  - For a 90° corner, the pricking should be at 45° from the vertex
  - For other angles, the pricking should bisect the angle
  - The distance along the bisector is calculated as: `half_width / sin(angle/2)`

  However, the tape edges are created by **offsetting the control path** perpendicular to the tangent.
  At sharp corners, these offset curves self-intersect. The **miter points** (where offset curves
  intersect) do NOT align with the angle bisectors.

  This creates a geometric conflict:
  - **Angle bisector approach**: Prickings are geometrically correct, but edges don't pass through them
  - **Miter point approach**: Edges and prickings align, but prickings aren't at angle bisectors

  **What Works Now**:
  - `miter_offset_edges()` (lines 655-673): Detects self-intersections and creates clean miter points
  - `calculate_angle_bisector()` (lines 615-653): Calculates geometric angle bisector direction
  - Exterior bisector calculated by negating interior bisector
  - Distance calculation accounts for corner angle

  **What Doesn't Work**:
  - When using angle bisectors for pricking positions, edges cross the tape interior
  - When using miter points for pricking positions, they're not at the correct geometric angles

  **Option 3 Status**: ATTEMPTED in Session 2 (see "Session 2: Tape Corner Angle Bisector Implementation" above)
  - Angle bisector pricking positions: ✅ WORKING
  - Edge reconstruction: ⚠️ PARTIAL - Creates artifacts, needs complete rewrite for smooth curves

  **Alternative Approaches Tried**:
  - ❌ Placing pricking on one edge only → causes edge crossing
  - ❌ Converging both edges to pricking point → tape width goes to zero at corners
  - ❌ Using centroid-based dot product to detect inward/outward → didn't work reliably
  - ❌ Testing both bisector directions and choosing based on distance to edges → still misaligned

  **Key Code Sections**:
  - Lines 777-779: Edge mitering (current approach)
  - Lines 781-832: Vertex pricking point generation (uses miter points currently)
  - Lines 615-653: `calculate_angle_bisector()` function (ready to use)
  - Lines 697-734: Path sampling with vertex detection

  **Test Case**: House-shaped closed path with 5 vertices:
  - Top peak (outside corner, ~45° angle)
  - Two top inside corners (~135° angles)
  - Two bottom corners (90° angles)

  **User Requirements** (confirmed 2025-12-07):
  - Top peak: Pricking must be directly above the control path vertex on outer edge
  - Bottom 90° corners: Pricking must be at 45° from vertex, on outer edge
  - Inside corners: Pricking must bisect the corner angle, on outer edge
  - **Edges must pass through the pricking points** (no gaps, no crossings)

#### 2. Tally (Rectangular)
- **Thread pairs**: Fixed at 2
- **Control shape**: Any shape (uses bounding box)
- **Features**:
  - Rectangle outline (no internal pricking)
  - Entry point (top center) and exit point (bottom center)
  - 4 corner connection markers (red circles)
  - Transform preservation for native rectangles
  - Connection tracking for entry/exit points
- **Location**: `create_tally_rect()` (lines 652-826)

#### 3. Tally (Leaf-shaped)
- **Thread pairs**: Fixed at 2
- **Control shape**: Ellipse (recommended), Rectangle, or any shape (paths use bounding box)
- **Features**:
  - Vesica piscis (lens) outline with pointed ends
  - Entry point (top) and exit point (bottom)
  - 2 connection markers at pointed ends
  - Mathematically accurate lens geometry
  - Full transform support for rotated/scaled ellipses and rectangles
  - Automatic orientation handling (horizontal and vertical shapes)
  - Group recursion (processes individual shapes within groups)
- **Location**: `create_tally_leaf()` (lines 850-1110)
- **Best practices**:
  - Use elongated shapes (aspect ratio ≥ 1.5:1) for proper lens shape
  - Works with ellipses (rx/ry) and rectangles (width/height)
  - Supports rotated ellipses, rotated rectangles, and grouped shapes
  - Warning shown if aspect ratio < 1.5
- **Transform handling**:
  - Ellipses: Extracts intrinsic rx/ry before applying transforms
  - Rectangles: Extracts intrinsic width/height before applying transforms
  - Paths: Falls back to bounding box (may not handle rotations well)

#### 4. Plait
- **Thread pairs**: Fixed at 2
- **Control shape**: Path (but renders as straight line)
- **Features**:
  - STRAIGHT line (braiding creates tension)
  - NO pricking along length (braided, not pinned)
  - Auto-snapping to nearby pricking points (configurable distance)
  - Connection tracking in metadata
  - Optional picots at midpoint (2 hollow circles + center pin)
  - Connection markers at start/end (red circles)
- **Location**: `create_plait()` (lines 1035-1212)

### 🚧 Partial Implementation

#### 5. Leaf
- **Thread pairs**: Currently fixed at 4, **will be variable** in future
- **Control shape**: Path (closed)
- **Features implemented**:
  - Closed outline with perimeter pricking
  - Central vein (dashed line from base to tip)
  - Entry/exit marker at base
  - Regular spacing of pricking points around perimeter
- **Location**: `create_leaf()` (lines 1213-1407)
- **Status**: Basic structure present, NOT YET FULLY TESTED

### 🔄 Core Infrastructure

#### Group Handling
- **Recursive element extraction**: `get_elements_recursively()` (lines 26-42)
  - Recursively descends into groups to find individual shapes
  - Prevents treating entire groups as single elements
  - Essential for processing multiple ellipses/shapes in groups

#### Connection System
- **Pricking point discovery**: `find_all_pricking_points()` (lines 128-194)
  - Scans all lace elements in document
  - Returns array of available connection points
  - Tracks active/inactive status

- **Nearest point snapping**: `find_nearest_pricking_point()` (lines 196-222)
  - Finds closest pricking within snap distance
  - Skips already-active points

- **Connection tracking**: `find_connections_to_control_path()` (lines 224-268)
  - Finds all plaits connected to an element
  - Uses stable control_path_id (not element_id)
  - Returns connection details (which end, which pricking)

#### Regeneration Support
- **Element lookup**: `find_lace_element_by_control_path()` (lines 107-126)
  - When regenerating, finds existing element by control path
  - Deletes old element before creating new one
  - Preserves connections across regenerations

#### Path Processing
- **Path sampling**: `sample_path()` (lines 295-411)
  - Dense sampling followed by arc-length parameterization
  - Regular spacing along curves
  - Returns (point, tangent, t_parameter) tuples

- **Bezier math**: Lines 476-502
  - Cubic bezier point/tangent calculation
  - Quadratic bezier support

- **Edge sampling**: `sample_edge_for_pricking()` (lines 424-474)
  - Equal intervals along polyline edges

## Thread Path Tracking (Future Implementation)

### Design Notes

**NOT YET IMPLEMENTED** - planned for after current element testing is complete.

#### Requirements
- Track **thread pairs** (not individual threads)
- Validate continuity and correctness
- Do NOT visualize paths on canvas
- Should validate that every pair has start and end points

#### Validation Rules
1. **Plait connections**: When a plait (2 pairs) connects to a tape, both pairs connect at that pricking point
2. **Tape continuity**: Pairs may join, be carried along the tape, and exit at different points
3. **Orphan detection**: Warn about elements with no connections
4. **Complete paths**: Validate that every thread pair has a start and end (tied onto pillow, tied off)
5. **Splits/joins**: Will be supported within tapes (metadata has `splits` array)

#### Existing Infrastructure
- `pricking_ports` array in tape metadata stores connection points
- `is_active` flags on pricking points
- Connection tracking between plaits and other elements
- Stable `control_path_id` for tracking across regenerations

## Settings

### User Parameters
- **Element type**: tape, tally_rect, tally_leaf, plait, leaf
- **Tape pairs**: 2-20 (default 8)
- **Tape width**: 1.0-50.0mm (default 10mm)
- **Plait picots**: boolean (default false)
- **Thread size**: 20-200 (default 60, higher = finer)
  - Affects pricking point spacing
  - Size 40 (coarse): ~0.4mm diameter
  - Size 60 (medium): ~0.25mm diameter
  - Size 80 (fine): ~0.2mm diameter
- **Show control path**: boolean (default true)
  - True: shows blue dashed control path at 50% opacity
  - False: hides control path completely
- **Snap distance**: 1.0-20.0mm (default 5mm)
  - Distance within which plaits snap to pricking points

### Spacing Calculation
Formula: `spacing = 2 × (30 / thread_size)` in mm
- Creates tight spacing for workers to pack closely
- Automatically adjusts for thread fineness

## Visual Elements

### Color Coding
- **Pricking points**:
  - Gray (#666666): Inactive/unconnected
  - Green (#00FF00): Connected to a plait
- **Connection markers**: Red outline circles (#FF0000)
- **Element outlines**: Black (#000000)
- **Control paths**: Blue dashed (#0000FF) when visible
- **Picots**: Black outline circles with gray center pin

### Sizes
- Pricking points: 0.8mm radius
- Connection markers: 1.5mm radius
- Element outlines: 0.5pt stroke width
- Picots: 0.6mm circles, 0.4mm center pin

## Metadata Structure

All lace elements store JSON metadata in `data-lace-metadata` attribute:

```json
{
  "element_type": "tape|tally_rect|tally_leaf|plait|leaf",
  "control_path_id": "path123",  // Stable ID for regeneration
  "pairs": 2,  // Thread pair count

  // Type-specific fields:

  // Tape:
  "initial_pairs": 8,
  "current_pairs": 8,
  "base_width_mm": 10.0,
  "splits": [],
  "pricking_ports": [
    {"point": [x, y], "edge": "left|right", "t": 0.5, "index": 0, "is_active": false}
  ],

  // Tally (both types):
  "width": 100.0,
  "height": 50.0,
  "entry_point": [x, y],
  "exit_point": [x, y],
  "connection_points": [[x, y], ...],

  // Plait:
  "has_picots": false,
  "start_point": [x, y],
  "end_point": [x, y],
  "connection_points": [[x, y], [x, y]],
  "start_connection": {...} | null,
  "end_connection": {...} | null,

  // Leaf:
  "pricking_points": [
    {"point": [x, y], "index": 0, "is_active": false}
  ],
  "entry_exit_point": [x, y]
}
```

## Current Testing Focus

### Completed Testing
- [x] Tally leaf geometry with rotated ellipses
- [x] Tally leaf positioning with grouped ellipses
- [x] Tally leaf horizontal orientation (rx > ry)
- [x] Group recursion for multiple ellipses

### Testing Areas (Remaining)
- [ ] Tape generation with various thread sizes
- [ ] Tape width scaling
- [x] Tally rectangle positioning - PASS (simple and path-converted rectangles)
- [x] Tally rectangle transforms - PASS (rotated rectangles work correctly)
- [x] Tally leaf all orientations - PASS (vertical, horizontal, rotated, grouped)
- [x] Plait rendering - PASS (straight lines, picots)
- [ ] Plait snapping behavior
- [ ] Connection detection (green highlighting)
- [ ] Regeneration stability
- [ ] Multiple element interactions

## Recent Fixes (2025-12-07)

### Session 1: Basic Element Testing

#### Rotated Rectangle Tally (Issue #1)
**Problem**: Tally rectangle appeared not to work on rotated rectangles
**Root Cause**: User error - rotated rectangle was not properly selected during testing
**Fix**: Added defensive check for rectangle tag anyway: `if not is_rect and control_path.tag.endswith('rect')`
**Location**: `create_tally_rect()` lines 1003-1008
**Status**: ✓ VERIFIED WORKING - Transform handling was correct all along. Additional check added as defense-in-depth.

#### Plait Error Message (Issue #2)
**Problem**: Error message "object needs to be converted to a path" appears even when plait renders correctly
**Root Cause**: User likely has multiple elements selected, including non-path elements. Extension processes all selected elements, shows error for non-paths but continues with valid paths.
**Fix**: Improved error message to clarify which element is being skipped and why
**Location**: `effect()` line 93
**Status**: Improved messaging (not a bug, working as designed)

### Session 2: Tape Corner Angle Bisector Implementation

#### Angle Bisector Pricking Positions (Option 3 - Partial Implementation)
**Goal**: Place vertex prickings at geometric angle bisectors as required by traditional bobbin lace
**Status**: ⚠️ PARTIALLY WORKING - Prickings appear at correct positions, but edges need refinement

**What Works**:
- ✅ `determine_path_winding()`: Detects CW/CCW path winding using shoelace formula
- ✅ `is_exterior_corner()`: Correctly identifies exterior vs interior corners using cross product
- ✅ `calculate_vertex_pricking_position()`: Calculates angle bisector positions with correct distance (`half_width / sin(angle/2)`)
- ✅ Centroid-based outer edge detection: Determines which edge (left/right) is farther from path center
- ✅ All 5 vertex prickings appear on house-shaped test path
- ✅ Prickings positioned at geometrically correct angle bisectors

**What Doesn't Work**:
- ❌ Edge reconstruction is oversimplified: Simple point replacement creates jagged edges
- ❌ Sharp triangular artifacts at some corners
- ❌ Edges are wavy/irregular due to dense sampling and point replacement
- ❌ Need smooth curve reconstruction, not just point substitution

**Technical Details**:
- **Lines 657-690**: `determine_path_winding()` - Uses shoelace formula to detect CW (-1) or CCW (1) winding
- **Lines 691-720**: `is_exterior_corner()` - Uses cross product; in SVG coords (Y-down): negative = exterior, positive = interior
- **Lines 722-774**: `calculate_vertex_pricking_position()` - Calculates position along angle bisector at distance `half_width / sin(half_angle)`
- **Lines 936-970**: Centroid-based edge selection - Measures distance from centroid to choose outer edge
- **Lines 972-994**: Edge reconstruction - **NEEDS COMPLETE REWRITE** for smooth results

**Current Edge Reconstruction Approach** (Simplified, causes artifacts):
```python
# For each vertex: replace offset point with angle bisector position
if vertex_pricking:
    if vertex_pricking['edge'] == 'left':
        left_edge_display.append(vertex_pricking['pricking_pos'])
    else:
        right_edge_display.append(vertex_pricking['pricking_pos'])
```

**Required for Full Solution** (Not Yet Implemented):
1. Smooth curve reconstruction around vertices
2. Proper tangent matching between segments
3. Possibly reduce sampling density
4. Handle edge curves properly, not just polylines
5. May need to use Bezier curves instead of polylines for edges

**User Requirements Verified**:
- Top peak: Pricking at angle bisector on outer edge ✅
- Bottom 90° corners: Prickings at 45° on outer edges ✅
- Inside corners: Prickings at angle bisectors on outer edges ✅
- Edges pass through prickings: ⚠️ Technically yes, but creates artifacts

**Next Steps** (Deferred):
- Complete rewrite of edge reconstruction for smooth curves
- Consider using Bezier path commands instead of polyline points
- Test with various corner angles and path shapes
- User education: Understanding how pricking relates to actual lace-making technique may inform better edge reconstruction approach

## Known Limitations

1. Leaf element not fully tested yet
2. Thread path validation not implemented
3. Dynamic tape width adjustment not implemented (pair count stays constant)
4. No split/join support in tapes yet
5. Connection tracking for tallies not fully implemented (always shows gray)
6. **Tally leaf with path shapes uses bounding box (may not handle rotations well)**
   - For shapes converted to paths, bounding boxes are axis-aligned
   - This may produce incorrect dimensions for rotated paths
   - **Recommended**: Use native Ellipse or Rectangle elements for leaf tallies
   - Both Ellipse and Rectangle properly extract intrinsic dimensions before applying transforms

## File Structure

- `bedfordshire_lace.py`: Main extension (1411 lines)
- `bedfordshire_lace.inx`: UI definition (43 lines)
- `DEVELOPMENT.md`: This file

## Next Steps

1. Complete testing of tapes, tallies, and plaits
2. Fix any issues discovered during testing
3. Implement thread path tracking system
4. Add validation for thread continuity
5. Complete leaf element testing
6. Implement dynamic tape width (pair splits/joins)

## Technical Notes

### Transform Handling
- **Ellipse transforms**: `composed_transform()` includes parent group transforms and element transform attribute, but NOT cx/cy translation
- **Transform composition**: For ellipses, final transform is: `composed_transform @ translate(cx,cy) @ rotate(90° if needed)`
- **Horizontal ellipses**: Automatically detect when rx > ry and add 90° rotation to match orientation
- **Group transforms**: Parent group transforms are included in `composed_transform()`

### Best Practices
- Use `get_elements_recursively()` to handle grouped shapes properly
- For ellipses, extract intrinsic rx/ry values to get true dimensions before rotation
- For rectangles, extract intrinsic width/height values to get true dimensions before rotation
- Bounding boxes on rotated shapes are axis-aligned (not useful for getting original dimensions)
- All coordinates should be in SVG user units (converted from mm as needed)
- Control paths use stable IDs to survive regeneration

### Development Focus
- User explicitly does NOT want to work on thread path tracking yet
- Focus is on ensuring current elements work correctly
- Tally leaf now fully tested and working with rotated/grouped ellipses
