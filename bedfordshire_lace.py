#!/usr/bin/env python3
"""
Bedfordshire Lace Pricking Pattern Generator for Inkscape
Generates pricking patterns (pin placement diagrams) for Bedfordshire bobbin lace.
"""

import inkex
from inkex import Path, PathElement, Group, Circle, Layer
import math
import json


class BedfordshireLace(inkex.EffectExtension):
    """Generate Bedfordshire lace pricking patterns from paths."""

    def add_arguments(self, pars):
        """Define command line arguments."""
        pars.add_argument("--element_type", type=str, default="tape", help="Type of lace element")
        pars.add_argument("--tape_pairs", type=int, default=8, help="Initial number of thread pairs for tape")
        pars.add_argument("--tape_width", type=float, default=10.0, help="Base width of tape in mm")
        pars.add_argument("--plait_picots", type=inkex.Boolean, default=False, help="Add picots to plait")
        pars.add_argument("--thread_size", type=int, default=60, help="Thread size (higher = finer)")
        pars.add_argument("--show_control_path", type=inkex.Boolean, default=True, help="Show control path")
        pars.add_argument("--snap_distance", type=float, default=5.0, help="Snap distance in mm")

    def get_elements_recursively(self, element):
        """
        Recursively extract all non-group elements from a selection.
        If element is a group, returns all children (recursively).
        Otherwise returns the element itself.
        """
        from inkex import Group

        if isinstance(element, Group):
            # Recursively get all children
            result = []
            for child in element:
                result.extend(self.get_elements_recursively(child))
            return result
        else:
            # Not a group, return the element itself
            return [element]

    def effect(self):
        """Main effect method called by Inkscape."""
        from inkex import Rectangle, Ellipse

        # Check if a path is selected
        if not self.svg.selected:
            inkex.errormsg("Please select a shape first.")
            return

        # Get or create the lace pattern layer
        lace_layer = self.get_or_create_layer("Lace Pattern")

        # Flatten selection (expand groups into individual elements)
        all_elements = []
        for element in self.svg.selected.values():
            all_elements.extend(self.get_elements_recursively(element))

        # Process ALL elements (including those from groups)
        for element in all_elements:
            element_id = element.get('id')

            # Check if this control path already has a lace element associated with it
            # If so, delete it before regenerating
            existing_element = self.find_lace_element_by_control_path(element_id)
            if existing_element is not None:
                # Remove the old element
                existing_element.getparent().remove(existing_element)

            # Route to appropriate element generator
            element_type = self.options.element_type

            if element_type == "tape":
                # Tape requires a path
                if not isinstance(element, PathElement):
                    inkex.errormsg(f"Tape requires a path object. Element '{element_id}' (type: {element.tag}) is not a path. Skipping this element. Please use the Bezier/pen tool or convert your shape to a path (Path > Object to Path).")
                    continue
                self.create_tape(element, lace_layer)

            elif element_type == "tally_rect":
                # Rectangular tally can use any shape (we just need bounding box)
                self.create_tally_rect(element, lace_layer)

            elif element_type == "tally_leaf":
                # Leaf tally can use any shape (we just need bounding box)
                self.create_tally_leaf(element, lace_layer)

            elif element_type == "plait":
                # Plait requires a path
                if not isinstance(element, PathElement):
                    inkex.errormsg(f"Plait requires a path object. Element '{element_id}' (type: {element.tag}) is not a path. Skipping this element. If this is not a path you intended to convert to a plait, please deselect it.")
                    continue
                self.create_plait(element, lace_layer)

            elif element_type == "leaf":
                # Leaf requires a path
                if not isinstance(element, PathElement):
                    inkex.errormsg(f"Leaf requires a path object. Element '{element_id}' is not a path. Please use the Bezier/pen tool or convert your shape to a path (Path > Object to Path).")
                    continue
                self.create_leaf(element, lace_layer)

        # Find the original drawing layer to unlock
        original_layer = None
        for layer in self.document.xpath('//svg:g[@inkscape:groupmode="layer"]', namespaces=inkex.NSS):
            label = layer.get(inkex.addNS('label', 'inkscape'))
            if label and 'lace' not in label.lower():
                original_layer = layer
                break

        # Unlock layers
        if original_layer is not None:
            original_layer.set(inkex.addNS('insensitive', 'sodipodi'), None)
        lace_layer.set(inkex.addNS('insensitive', 'sodipodi'), None)

    def get_or_create_layer(self, layer_name):
        """Get existing layer or create new one."""
        # Search for existing layer
        for layer in self.document.xpath('//svg:g[@inkscape:groupmode="layer"]', namespaces=inkex.NSS):
            if layer.get(inkex.addNS('label', 'inkscape')) == layer_name:
                return layer

        # Create new layer if not found
        layer = Layer()
        layer.set(inkex.addNS('label', 'inkscape'), layer_name)
        self.document.getroot().append(layer)
        return layer

    def find_lace_element_by_control_path(self, control_path_id):
        """
        Find a lace element (tape, tally, plait, etc.) that was created from the given control path.

        Returns the element group if found, None otherwise.
        """
        # Search all groups in the document
        for group in self.document.xpath('//svg:g', namespaces=inkex.NSS):
            # Check if this group has lace metadata
            metadata_str = group.get('data-lace-metadata')
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                    # Check if this element references our control path
                    if metadata.get('control_path_id') == control_path_id:
                        return group
                except (json.JSONDecodeError, KeyError):
                    continue

        return None

    def find_all_pricking_points(self):
        """
        Find all pricking points from all lace elements in the document.

        Returns list of dicts with:
        - point: (x, y) tuple
        - element_id: ID of the element this pricking belongs to
        - control_path_id: ID of the control path (stable across regenerations)
        - element_type: 'tape', 'tally_rect', etc.
        - pricking_index: index in the element's pricking list
        - is_active: whether already connected
        """
        all_prickings = []

        # Search all groups in the document
        for group in self.document.xpath('//svg:g', namespaces=inkex.NSS):
            metadata_str = group.get('data-lace-metadata')
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                    element_type = metadata.get('element_type')
                    element_id = group.get('id')
                    control_path_id = metadata.get('control_path_id')

                    # Get pricking points or connection points depending on element type
                    if element_type == 'tape':
                        # Tape has pricking_ports array
                        pricking_ports = metadata.get('pricking_ports', [])
                        for i, port in enumerate(pricking_ports):
                            all_prickings.append({
                                'point': tuple(port['point']),
                                'element_id': element_id,
                                'control_path_id': control_path_id,
                                'element_type': element_type,
                                'pricking_index': i,
                                'is_active': port.get('is_active', False)
                            })

                    elif element_type in ['tally_rect', 'tally_leaf']:
                        # Tallies have entry_point and exit_point
                        entry = metadata.get('entry_point')
                        exit_pt = metadata.get('exit_point')

                        if entry:
                            all_prickings.append({
                                'point': tuple(entry),
                                'element_id': element_id,
                                'control_path_id': control_path_id,
                                'element_type': element_type,
                                'pricking_index': 0,  # Entry
                                'is_active': False  # TODO: track this
                            })

                        if exit_pt:
                            all_prickings.append({
                                'point': tuple(exit_pt),
                                'element_id': element_id,
                                'control_path_id': control_path_id,
                                'element_type': element_type,
                                'pricking_index': 1,  # Exit
                                'is_active': False  # TODO: track this
                            })

                except (json.JSONDecodeError, KeyError):
                    continue

        return all_prickings

    def find_nearest_pricking_point(self, target_point, snap_distance):
        """
        Find the nearest pricking point to the target point within snap distance.

        Returns dict with pricking info if found, None otherwise.
        """
        all_prickings = self.find_all_pricking_points()

        nearest = None
        nearest_dist = snap_distance

        for pricking in all_prickings:
            # Skip already active pricking points
            if pricking['is_active']:
                continue

            # Calculate distance
            dist = math.hypot(
                pricking['point'][0] - target_point[0],
                pricking['point'][1] - target_point[1]
            )

            if dist < nearest_dist:
                nearest_dist = dist
                nearest = pricking

        return nearest

    def find_connections_to_control_path(self, control_path_id):
        """
        Find all plaits that connect to a specific control path.

        Uses control_path_id instead of element_id for stability across regenerations.

        Returns list of dicts with:
        - plait_id: ID of the plait
        - connected_at: 'start' or 'end'
        - pricking_index: which pricking point on the element
        """
        connections = []

        # Search all groups for plaits
        for group in self.document.xpath('//svg:g', namespaces=inkex.NSS):
            metadata_str = group.get('data-lace-metadata')
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)

                    if metadata.get('element_type') == 'plait':
                        plait_id = group.get('id')

                        # Check start connection
                        start_conn = metadata.get('start_connection')
                        if start_conn and start_conn.get('control_path_id') == control_path_id:
                            connections.append({
                                'plait_id': plait_id,
                                'connected_at': 'start',
                                'pricking_index': start_conn.get('pricking_index')
                            })

                        # Check end connection
                        end_conn = metadata.get('end_connection')
                        if end_conn and end_conn.get('control_path_id') == control_path_id:
                            connections.append({
                                'plait_id': plait_id,
                                'connected_at': 'end',
                                'pricking_index': end_conn.get('pricking_index')
                            })

                except (json.JSONDecodeError, KeyError):
                    continue

        return connections

    def mm_to_units(self, mm):
        """Convert millimeters to SVG user units."""
        return mm * self.svg.unittouu('1mm')

    def calculate_spacing(self):
        """
        Calculate pricking point spacing based on thread size.

        Thread size is inversely proportional to diameter:
        - Size 40 (coarse): ~0.4mm diameter
        - Size 60 (medium): ~0.25mm diameter
        - Size 80 (fine): ~0.2mm diameter
        - Size 120 (very fine): ~0.15mm diameter

        Spacing should be tight enough that workers pack closely in straight sections.
        Use approximately 2x thread diameter to allow thread to pass.
        """
        # Approximate thread diameter in mm
        thread_diameter_mm = 30.0 / self.options.thread_size

        # Spacing is about 2x diameter (allows thread to pass comfortably)
        spacing_mm = thread_diameter_mm * 2.0

        return self.mm_to_units(spacing_mm)

    def sample_path(self, path_data, spacing):
        """
        Sample a path at regular intervals.
        Returns tuple: (samples, vertices)
        - samples: list of (point, tangent, t_parameter, is_vertex) tuples
        - vertices: list of actual path vertex points in order

        Uses dense sampling followed by resampling for accuracy.
        Ensures all path vertices (corners) are included in the samples.
        """
        path = Path(path_data)

        # Build segments with proper coordinate extraction using superpath
        segments = []
        vertices = []  # Track actual path vertices
        subpaths = list(path.to_absolute().to_superpath())

        if not subpaths:
            return []

        # Process first subpath only
        subpath = subpaths[0]

        for i in range(len(subpath) - 1):
            # Each item in subpath is [[x,y], [x,y], [x,y]]
            # representing [handle_before, point, handle_after]
            p0 = subpath[i][1]  # Current endpoint
            p1 = subpath[i][2]  # First control point (handle after current)
            p2 = subpath[i+1][0]  # Second control point (handle before next)
            p3 = subpath[i+1][1]  # Next endpoint

            segments.append((p0, p1, p2, p3))

            # Track ALL vertices (path endpoints between segments)
            # These are the points where the path direction changes
            if i == 0:
                vertices.append(tuple(p0))
            # Always include the endpoint of each segment as a potential vertex
            vertices.append(tuple(p3))

        if not segments:
            return []

        # Step 1: Create a VERY dense sampling of the path (many points)
        dense_points = []
        dense_tangents = []

        for seg in segments:
            # Sample each bezier segment with 100 points for accuracy
            for i in range(100):
                t = i / 100
                pt = self.cubic_bezier_point(seg[0], seg[1], seg[2], seg[3], t)
                tangent = self.cubic_bezier_tangent(seg[0], seg[1], seg[2], seg[3], t)
                dense_points.append(pt)
                dense_tangents.append(tangent)

        # Add the final point
        seg = segments[-1]
        pt = self.cubic_bezier_point(seg[0], seg[1], seg[2], seg[3], 1.0)
        tangent = self.cubic_bezier_tangent(seg[0], seg[1], seg[2], seg[3], 1.0)
        dense_points.append(pt)
        dense_tangents.append(tangent)

        # Step 2: Calculate cumulative arc length along dense points
        cumulative_dist = [0]
        for i in range(1, len(dense_points)):
            dist = math.hypot(
                dense_points[i][0] - dense_points[i-1][0],
                dense_points[i][1] - dense_points[i-1][1]
            )
            cumulative_dist.append(cumulative_dist[-1] + dist)

        total_length = cumulative_dist[-1]
        if total_length == 0:
            return []

        # Step 3: Resample at EXACT regular intervals
        samples = []
        target_dist = spacing

        while target_dist < total_length - spacing * 0.1:
            # Find the two dense points that bracket this distance
            idx = 0
            for i in range(len(cumulative_dist) - 1):
                if cumulative_dist[i] <= target_dist <= cumulative_dist[i + 1]:
                    idx = i
                    break

            # Interpolate between dense_points[idx] and dense_points[idx+1]
            if idx < len(dense_points) - 1:
                segment_length = cumulative_dist[idx + 1] - cumulative_dist[idx]
                if segment_length > 0:
                    t_local = (target_dist - cumulative_dist[idx]) / segment_length
                else:
                    t_local = 0

                p1 = dense_points[idx]
                p2 = dense_points[idx + 1]

                # Interpolate position
                point = (
                    p1[0] + t_local * (p2[0] - p1[0]),
                    p1[1] + t_local * (p2[1] - p1[1])
                )

                # Interpolate tangent
                t1 = dense_tangents[idx]
                t2 = dense_tangents[idx + 1]
                tangent_vec = (
                    t1[0] + t_local * (t2[0] - t1[0]),
                    t1[1] + t_local * (t2[1] - t1[1])
                )

                # Normalize tangent
                tangent_len = math.hypot(tangent_vec[0], tangent_vec[1])
                if tangent_len > 0:
                    tangent = (tangent_vec[0] / tangent_len, tangent_vec[1] / tangent_len)
                else:
                    tangent = (1, 0)

                # Global t parameter
                t_param = target_dist / total_length

                samples.append((point, tangent, t_param))

            target_dist += spacing

        # Step 4: Add vertices to ensure corners are always included
        # Find the position of each vertex in the dense sampling
        vertex_samples = []
        for vertex in vertices:
            # Find the closest point in dense_points to this vertex
            min_dist = float('inf')
            closest_idx = 0
            for i, dp in enumerate(dense_points):
                dist = math.hypot(dp[0] - vertex[0], dp[1] - vertex[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            # Get the tangent at this vertex
            if closest_idx < len(dense_tangents):
                tangent = dense_tangents[closest_idx]
                # Normalize
                tangent_len = math.hypot(tangent[0], tangent[1])
                if tangent_len > 0:
                    tangent = (tangent[0] / tangent_len, tangent[1] / tangent_len)
                else:
                    tangent = (1, 0)
            else:
                tangent = (1, 0)

            # Get t parameter
            t_param = cumulative_dist[closest_idx] / total_length if total_length > 0 else 0

            vertex_samples.append((vertex, tangent, t_param, cumulative_dist[closest_idx]))

        # Step 5: Merge vertex samples with regular samples, sorted by distance along path
        # Add distance to regular samples for sorting and mark which are vertices
        samples_with_dist = [(s[0], s[1], s[2], s[2] * total_length, False) for s in samples]
        vertex_samples_marked = [(v[0], v[1], v[2], v[3], True) for v in vertex_samples]

        # Combine both lists
        all_samples = samples_with_dist + vertex_samples_marked

        # Sort by distance along path
        all_samples.sort(key=lambda x: x[3])

        # Remove duplicates that are very close together (within 0.1 * spacing)
        # Keep vertex samples over regular samples when deduplicating
        filtered_samples = []
        last_dist = -spacing
        for sample in all_samples:
            if sample[3] - last_dist > spacing * 0.1:
                filtered_samples.append((sample[0], sample[1], sample[2], sample[4]))  # Include is_vertex flag
                last_dist = sample[3]
            elif sample[4]:  # If this is a vertex and we're within dedup distance, replace previous
                if filtered_samples:
                    filtered_samples[-1] = (sample[0], sample[1], sample[2], True)

        return filtered_samples, vertices

    def approximate_bezier_length(self, p0, p1, p2, p3, steps=20):
        """Approximate bezier curve length by sampling."""
        length = 0
        prev_pt = p0
        for i in range(1, steps + 1):
            t = i / steps
            pt = self.cubic_bezier_point(p0, p1, p2, p3, t)
            length += math.hypot(pt[0] - prev_pt[0], pt[1] - prev_pt[1])
            prev_pt = pt
        return length

    def sample_edge_for_pricking(self, edge_points, target_spacing):
        """
        Sample points along an edge at equal intervals.
        edge_points: list of (x, y) tuples forming a polyline
        target_spacing: desired spacing between samples
        Returns: list of (x, y) tuples at equal intervals
        """
        if len(edge_points) < 2:
            return []

        # Calculate cumulative distances along the edge
        cumulative_dist = [0]
        for i in range(1, len(edge_points)):
            dist = math.hypot(
                edge_points[i][0] - edge_points[i-1][0],
                edge_points[i][1] - edge_points[i-1][1]
            )
            cumulative_dist.append(cumulative_dist[-1] + dist)

        total_length = cumulative_dist[-1]
        if total_length == 0:
            return []

        # Sample at regular intervals
        samples = []
        target_dist = target_spacing

        while target_dist < total_length - target_spacing * 0.1:
            # Find which segment this distance falls in
            for i in range(len(cumulative_dist) - 1):
                if cumulative_dist[i] <= target_dist < cumulative_dist[i + 1]:
                    # Interpolate between edge_points[i] and edge_points[i+1]
                    segment_length = cumulative_dist[i + 1] - cumulative_dist[i]
                    if segment_length > 0:
                        t = (target_dist - cumulative_dist[i]) / segment_length
                    else:
                        t = 0

                    p1 = edge_points[i]
                    p2 = edge_points[i + 1]

                    sample_pt = (
                        p1[0] + t * (p2[0] - p1[0]),
                        p1[1] + t * (p2[1] - p1[1])
                    )
                    samples.append(sample_pt)
                    break

            target_dist += target_spacing

        return samples

    def cubic_bezier_point(self, p0, p1, p2, p3, t):
        """Calculate point on cubic bezier curve at parameter t."""
        s = 1 - t
        x = s**3 * p0[0] + 3 * s**2 * t * p1[0] + 3 * s * t**2 * p2[0] + t**3 * p3[0]
        y = s**3 * p0[1] + 3 * s**2 * t * p1[1] + 3 * s * t**2 * p2[1] + t**3 * p3[1]
        return (x, y)

    def cubic_bezier_tangent(self, p0, p1, p2, p3, t):
        """Calculate tangent vector on cubic bezier curve at parameter t."""
        s = 1 - t
        dx = 3 * s**2 * (p1[0] - p0[0]) + 6 * s * t * (p2[0] - p1[0]) + 3 * t**2 * (p3[0] - p2[0])
        dy = 3 * s**2 * (p1[1] - p0[1]) + 6 * s * t * (p2[1] - p1[1]) + 3 * t**2 * (p3[1] - p2[1])
        return (dx, dy)

    def quadratic_bezier_point(self, p0, p1, p2, t):
        """Calculate point on quadratic bezier curve at parameter t."""
        s = 1 - t
        x = s**2 * p0[0] + 2 * s * t * p1[0] + t**2 * p2[0]
        y = s**2 * p0[1] + 2 * s * t * p1[1] + t**2 * p2[1]
        return (x, y)

    def quadratic_bezier_tangent(self, p0, p1, p2, t):
        """Calculate tangent vector on quadratic bezier curve at parameter t."""
        s = 1 - t
        dx = 2 * s * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0])
        dy = 2 * s * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1])
        return (dx, dy)

    def line_intersection(self, p1, p2, p3, p4):
        """
        Find intersection point of two line segments.
        p1-p2 is first segment, p3-p4 is second segment.
        Returns intersection point (x, y) or None if lines don't intersect.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # Lines are parallel

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Check if intersection is within both segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)

        return None

    def calculate_angle_bisector(self, prev_point, vertex, next_point):
        """
        Calculate the angle bisector direction at a vertex.

        Args:
            prev_point: Point before the vertex
            vertex: The vertex point
            next_point: Point after the vertex

        Returns:
            Normalized direction vector of the angle bisector
        """
        # Vector from vertex to previous point
        v1 = (prev_point[0] - vertex[0], prev_point[1] - vertex[1])
        v1_len = math.hypot(v1[0], v1[1])
        if v1_len > 0:
            v1 = (v1[0] / v1_len, v1[1] / v1_len)
        else:
            v1 = (0, 0)

        # Vector from vertex to next point
        v2 = (next_point[0] - vertex[0], next_point[1] - vertex[1])
        v2_len = math.hypot(v2[0], v2[1])
        if v2_len > 0:
            v2 = (v2[0] / v2_len, v2[1] / v2_len)
        else:
            v2 = (0, 0)

        # Bisector is the normalized sum of the two unit vectors
        bisector = (v1[0] + v2[0], v1[1] + v2[1])
        bisector_len = math.hypot(bisector[0], bisector[1])

        if bisector_len > 0:
            bisector = (bisector[0] / bisector_len, bisector[1] / bisector_len)
        else:
            # Vectors are opposite - use perpendicular
            bisector = (-v1[1], v1[0])

        return bisector

    def is_exterior_corner(self, prev_point, vertex, next_point):
        """
        Determine if a corner is exterior (convex) or interior (concave).

        Uses the cross product to determine the turning direction.
        For a clockwise-wound closed path, left turns are exterior corners.

        Args:
            prev_point: Point before the vertex
            vertex: The vertex point
            next_point: Point after the vertex

        Returns:
            True if exterior corner, False if interior corner
        """
        # Vectors
        v1 = (vertex[0] - prev_point[0], vertex[1] - prev_point[1])
        v2 = (next_point[0] - vertex[0], next_point[1] - vertex[1])

        # Cross product (z-component)
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        # Positive cross product = left turn = exterior (for counter-clockwise paths)
        # SVG typically uses counter-clockwise winding for positive areas
        return cross > 0

    def calculate_vertex_pricking_position(self, prev_point, vertex, next_point, half_width, is_exterior):
        """
        Calculate the pricking point position at a vertex using angle bisector geometry.

        For bobbin lace, pricking points at corners must be positioned along the angle bisector
        of the corner. The distance along the bisector depends on the corner angle.

        Args:
            prev_point: Point before the vertex
            vertex: The vertex point
            next_point: Point after the vertex
            half_width: Half the tape width
            is_exterior: True if this is an exterior corner (outside edge), False for interior

        Returns:
            (x, y) tuple of the pricking point position
        """
        # Get interior bisector direction
        interior_bisector = self.calculate_angle_bisector(prev_point, vertex, next_point)

        # For exterior corners, negate the bisector
        if is_exterior:
            bisector = (-interior_bisector[0], -interior_bisector[1])
        else:
            bisector = interior_bisector

        # Calculate the angle between the two edges
        v1 = (prev_point[0] - vertex[0], prev_point[1] - vertex[1])
        v2 = (next_point[0] - vertex[0], next_point[1] - vertex[1])

        v1_len = math.hypot(v1[0], v1[1])
        v2_len = math.hypot(v2[0], v2[1])

        if v1_len > 0 and v2_len > 0:
            v1_norm = (v1[0] / v1_len, v1[1] / v1_len)
            v2_norm = (v2[0] / v2_len, v2[1] / v2_len)

            # Dot product gives cos(angle)
            dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
            # Clamp to [-1, 1] to handle floating point errors
            dot_product = max(-1.0, min(1.0, dot_product))

            angle_rad = math.acos(dot_product)
            half_angle = angle_rad / 2

            # Distance along bisector = half_width / sin(half_angle)
            # For very small angles, use a minimum to avoid infinity
            if abs(math.sin(half_angle)) > 0.01:
                distance = half_width / math.sin(half_angle)
            else:
                distance = half_width * 10  # Cap at reasonable value
        else:
            # Degenerate case - use half_width
            distance = half_width

        # Position pricking along bisector
        pricking_x = vertex[0] + bisector[0] * distance
        pricking_y = vertex[1] + bisector[1] * distance

        return (pricking_x, pricking_y)

    def miter_offset_edges(self, edge_points):
        """
        Fix self-intersecting offset curves by detecting sharp turns.
        At sharp corners, finds where edges cross and replaces crossing segments with miter point.
        Returns tuple: (cleaned edge points list, list of miter point info dicts)
        """
        if len(edge_points) < 3:
            return edge_points, []

        mitered_points = []
        miter_info = []  # Track which indices were mitered and what the miter point is
        skip_until = -1

        for i in range(len(edge_points)):
            # If we're skipping points due to a previous miter, continue
            if i < skip_until:
                continue

            # At corners, look ahead to find where this edge crosses a future edge
            if i < len(edge_points) - 3:
                # Try to find intersection with future segments
                intersection_found = False

                # Look ahead progressively further
                for lookahead in range(2, min(15, len(edge_points) - i)):
                    j = i + lookahead
                    if j >= len(edge_points):
                        break

                    # Check if edge from i to i+1 intersects with edge from j-1 to j
                    if i + 1 < len(edge_points) and j < len(edge_points):
                        intersection = self.line_intersection(
                            edge_points[i],
                            edge_points[i + 1],
                            edge_points[j - 1],
                            edge_points[j]
                        )

                        if intersection:
                            # Found intersection - this is a corner that needs mitering
                            mitered_points.append(intersection)
                            # Record miter info: which range of original indices map to this miter
                            miter_info.append({
                                'point': intersection,
                                'start_idx': i,
                                'end_idx': j - 1,
                                'mitered_idx': len(mitered_points) - 1
                            })
                            skip_until = j
                            intersection_found = True
                            break

                if intersection_found:
                    continue

            # No intersection, add the point normally
            mitered_points.append(edge_points[i])

        return mitered_points, miter_info

    def create_tape(self, control_path, layer):
        """Create a tape element from the selected path."""
        # Get parameters
        initial_pairs = self.options.tape_pairs
        base_width_mm = self.options.tape_width
        base_width = self.mm_to_units(base_width_mm)
        spacing = self.calculate_spacing()

        # Create a group for this tape
        tape_group = Group()
        tape_group.set('id', self.svg.get_unique_id('tape_'))

        # Check if path is closed by looking for 'Z' or 'z' command in path data
        path_data = control_path.get('d')
        is_closed = 'z' in path_data.lower()

        # Sample the path
        samples, path_vertices = self.sample_path(path_data, spacing)

        if len(samples) < 2:
            inkex.errormsg("Path is too short to generate tape.")
            return

        # For now, fixed pair count (dynamic adjustment comes later)
        current_pairs = initial_pairs
        width = base_width * (current_pairs / initial_pairs)

        # Generate edge points AND track vertex information from samples
        # Keep these separate - pricking points don't get mitered
        left_edge_raw = []
        right_edge_raw = []
        sample_info = []  # Track which samples are vertices
        alternating = True

        for i, (point, tangent, t_param, is_vertex) in enumerate(samples):
            # Calculate perpendicular (normal) to the tangent
            perp = (-tangent[1], tangent[0])

            # Calculate edge points
            half_width = width / 2
            left_point = (point[0] + perp[0] * half_width, point[1] + perp[1] * half_width)
            right_point = (point[0] - perp[0] * half_width, point[1] - perp[1] * half_width)

            left_edge_raw.append(left_point)
            right_edge_raw.append(right_point)

            # Track sample info
            sample_info.append({
                'index': i,
                'center': point,
                'tangent': tangent,
                'left': left_point,
                'right': right_point,
                'edge_for_pricking': 'right' if alternating else 'left',
                't': t_param,
                'is_vertex': is_vertex
            })

            alternating = not alternating

        # NEW APPROACH: Calculate angle bisector pricking positions for vertices
        # and reconstruct edges to pass through them

        # First, identify vertices and calculate their pricking positions
        vertex_info = []  # Store vertex pricking information

        for i, info in enumerate(sample_info):
            if info['is_vertex']:
                # Find prev and next path vertices for angle calculation
                idx = info['index']

                # Find indices in path_vertices
                vertex_idx = -1
                for vi, v in enumerate(path_vertices):
                    # Check if this vertex matches our sample point
                    if math.hypot(v[0] - info['center'][0], v[1] - info['center'][1]) < 0.01:
                        vertex_idx = vi
                        break

                if vertex_idx >= 0 and len(path_vertices) >= 3:
                    # Get prev and next vertices (with wraparound for closed paths)
                    prev_v_idx = (vertex_idx - 1) % len(path_vertices)
                    next_v_idx = (vertex_idx + 1) % len(path_vertices)

                    prev_vertex = path_vertices[prev_v_idx]
                    curr_vertex = path_vertices[vertex_idx]
                    next_vertex = path_vertices[next_v_idx]

                    # Determine if this is an exterior or interior corner
                    is_exterior = self.is_exterior_corner(prev_vertex, curr_vertex, next_vertex)

                    # ALL prickings go on the outer edge of the tape
                    # For counter-clockwise wound paths, the left offset is the outer edge
                    edge_for_pricking = 'left'

                    # Calculate angle bisector pricking position
                    # For exterior corners: bisector points outward (away from shape interior)
                    # For interior corners: bisector points inward (into the notch)
                    # Both should position the pricking on the outer (left) edge
                    pricking_pos = self.calculate_vertex_pricking_position(
                        prev_vertex, curr_vertex, next_vertex, half_width, is_exterior
                    )

                    vertex_info.append({
                        'sample_idx': idx,
                        'vertex_idx': vertex_idx,
                        'pricking_pos': pricking_pos,
                        'edge': edge_for_pricking,
                        'is_exterior': is_exterior,
                        't': info['t']
                    })

        # Reconstruct edges to pass through angle bisector pricking points
        # Strategy: Replace the offset point at each vertex with the angle bisector position
        left_edge_reconstructed = []
        right_edge_reconstructed = []

        for i, info in enumerate(sample_info):
            # Check if this sample is a vertex with a pricking
            vertex_pricking = None
            for v_info in vertex_info:
                if v_info['sample_idx'] == info['index']:
                    vertex_pricking = v_info
                    break

            if vertex_pricking:
                # Use the angle bisector pricking position for the appropriate edge
                if vertex_pricking['edge'] == 'left':
                    left_edge_reconstructed.append(vertex_pricking['pricking_pos'])
                    right_edge_reconstructed.append(info['right'])
                else:
                    left_edge_reconstructed.append(info['left'])
                    right_edge_reconstructed.append(vertex_pricking['pricking_pos'])
            else:
                # Regular sample - use normal offsets
                left_edge_reconstructed.append(info['left'])
                right_edge_reconstructed.append(info['right'])

        # Use reconstructed edges for display
        left_edge_display = left_edge_reconstructed
        right_edge_display = right_edge_reconstructed

        # Build pricking points
        pricking_points = []
        vertex_prickings = []

        for info in sample_info:
            idx = info['index']
            edge = info['edge_for_pricking']
            is_vertex = info['is_vertex']

            # Check if this is a vertex with angle bisector pricking
            vertex_pricking = None
            for v_info in vertex_info:
                if v_info['sample_idx'] == idx:
                    vertex_pricking = v_info
                    break

            if vertex_pricking:
                # Use angle bisector pricking position
                vertex_prickings.append({
                    'point': vertex_pricking['pricking_pos'],
                    'edge': vertex_pricking['edge'],
                    't': vertex_pricking['t'],
                    'index': idx,
                    'is_active': False,
                    'is_vertex': True
                })
            elif not is_vertex:
                # Regular pricking point (not at a vertex)
                pricking_point = info['left'] if edge == 'left' else info['right']
                pricking_points.append({
                    'point': pricking_point,
                    'edge': edge,
                    't': info['t'],
                    'index': idx,
                    'is_active': False
                })

        # Deduplicate vertex prickings (in case multiple vertices are very close)
        deduplicated_vertices = []
        for vertex_pricking in vertex_prickings:
            is_duplicate = False
            for existing in deduplicated_vertices:
                dist = math.hypot(
                    vertex_pricking['point'][0] - existing['point'][0],
                    vertex_pricking['point'][1] - existing['point'][1]
                )
                if dist < self.mm_to_units(0.5):
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated_vertices.append(vertex_pricking)

        # Add all vertex prickings
        pricking_points.extend(deduplicated_vertices)

        # For closed paths, add first point at end to complete the loop
        if is_closed:
            if len(left_edge_display) > 0:
                left_edge_display.append(left_edge_display[0])
            if len(right_edge_display) > 0:
                right_edge_display.append(right_edge_display[0])

        # Use display edges for rendering
        left_edge = left_edge_display
        right_edge = right_edge_display

        # Draw the tape edges
        if is_closed:
            # For closed paths, draw two separate closed edge paths
            # This avoids crossing lines at corners

            # Left edge (closed loop)
            left_path = PathElement()
            left_commands = [f"M {left_edge[0][0]},{left_edge[0][1]}"]
            for p in left_edge[1:]:
                left_commands.append(f"L {p[0]},{p[1]}")
            left_commands.append("Z")
            left_path.set('d', ' '.join(left_commands))
            left_path.style = {
                'fill': 'none',
                'stroke': '#000000',
                'stroke-width': '0.5'
            }
            tape_group.append(left_path)

            # Right edge (closed loop)
            right_path = PathElement()
            right_commands = [f"M {right_edge[0][0]},{right_edge[0][1]}"]
            for p in right_edge[1:]:
                right_commands.append(f"L {p[0]},{p[1]}")
            right_commands.append("Z")
            right_path.set('d', ' '.join(right_commands))
            right_path.style = {
                'fill': 'none',
                'stroke': '#000000',
                'stroke-width': '0.5'
            }
            tape_group.append(right_path)
        else:
            # For open paths, draw two separate edge lines
            # Left edge
            left_path = PathElement()
            left_commands = [f"M {left_edge[0][0]},{left_edge[0][1]}"]
            for p in left_edge[1:]:
                left_commands.append(f"L {p[0]},{p[1]}")
            left_path.set('d', ' '.join(left_commands))
            left_path.style = {
                'fill': 'none',
                'stroke': '#000000',
                'stroke-width': '0.5'
            }
            tape_group.append(left_path)

            # Right edge
            right_path = PathElement()
            right_commands = [f"M {right_edge[0][0]},{right_edge[0][1]}"]
            for p in right_edge[1:]:
                right_commands.append(f"L {p[0]},{p[1]}")
            right_path.set('d', ' '.join(right_commands))
            right_path.style = {
                'fill': 'none',
                'stroke': '#000000',
                'stroke-width': '0.5'
            }
            tape_group.append(right_path)

        # Check which pricking points are connected to plaits
        # Use control_path_id for stability across regenerations
        control_path_id = control_path.get('id')

        # Find all connections to this tape's control path
        connections = self.find_connections_to_control_path(control_path_id)
        connected_indices = set()
        for conn in connections:
            connected_indices.add(conn['pricking_index'])

        # Draw pricking points (gray for inactive, green for connected)
        # Size them appropriately - visible but not overwhelming
        point_radius = self.mm_to_units(0.8)  # 0.8mm radius for visibility
        for pp in pricking_points:
            circle = Circle()
            circle.set('cx', str(pp['point'][0]))
            circle.set('cy', str(pp['point'][1]))
            circle.set('r', str(point_radius))

            # Check if this pricking point is connected
            is_connected = pp['index'] in connected_indices

            circle.style = {
                'fill': '#00FF00' if is_connected else '#666666',  # Green if connected, gray otherwise
                'stroke': 'none'
            }
            tape_group.append(circle)

        # Store metadata
        metadata = {
            'element_type': 'tape',
            'initial_pairs': initial_pairs,
            'current_pairs': current_pairs,
            'base_width_mm': base_width_mm,
            'splits': [],
            'pricking_ports': pricking_points,
            'control_path_id': control_path.get('id')
        }

        tape_group.set('data-lace-metadata', json.dumps(metadata))

        # Add to layer
        layer.append(tape_group)

        # Style the control path to make it clearly distinct
        if self.options.show_control_path:
            # Make it visible but clearly different
            control_path.style['opacity'] = '0.5'
            control_path.style['stroke'] = '#0000FF'  # Blue
            control_path.style['stroke-dasharray'] = '5,5'
            control_path.style['fill'] = 'none'
        else:
            # Hide it completely
            control_path.style['display'] = 'none'

    def create_tally_rect(self, control_path, layer):
        """
        Create a rectangular tally element.

        Tally is a solid woven area with:
        - Rectangle outline (no internal pricking)
        - Pricking points only at entry and exit (top and bottom)
        - Connection markers at all 4 corners
        - Fixed at 2 thread pairs
        """
        from inkex import Rectangle as RectElement

        # Check if this is an actual Rectangle element (not a path)
        # Also check for rectangle attributes in case the element has a transform
        use_transform = False
        is_rect = isinstance(control_path, RectElement)

        # Also check if element has rectangle attributes (x, y, width, height)
        # even if it's not strictly a RectElement type
        if not is_rect and control_path.tag.endswith('rect'):
            is_rect = True

        if is_rect:
            # Use original dimensions from the rectangle
            use_transform = True
            x = float(control_path.get('x') or '0')
            y = float(control_path.get('y') or '0')
            width = float(control_path.get('width') or '0')
            height = float(control_path.get('height') or '0')

            x_min = x
            y_min = y
            x_max = x + width
            y_max = y + height

            # Calculate positions in untransformed space
            top_center = (x + width / 2, y)
            bottom_center = (x + width / 2, y + height)
            corners = [
                (x, y),              # Top-left
                (x + width, y),      # Top-right
                (x, y + height),     # Bottom-left
                (x + width, y + height),  # Bottom-right
            ]
        else:
            # Get bounding box for paths or other shapes
            bbox = control_path.bounding_box()

            if bbox is None:
                inkex.errormsg("Could not get bounding box of selected shape.")
                return

            # Extract dimensions from transformed bounding box
            x_min, x_max = bbox.left, bbox.right
            y_min, y_max = bbox.top, bbox.bottom
            width = x_max - x_min
            height = y_max - y_min

            top_center = (x_min + width / 2, y_min)
            bottom_center = (x_min + width / 2, y_max)
            corners = [
                (x_min, y_min),  # Top-left
                (x_max, y_min),  # Top-right
                (x_min, y_max),  # Bottom-left
                (x_max, y_max),  # Bottom-right
            ]

        # Create a group for this tally
        tally_group = Group()
        tally_group.set('id', self.svg.get_unique_id('tally_rect_'))

        # Copy transform if we're using original dimensions (rectangles)
        if use_transform:
            transform = control_path.get('transform')
            if transform:
                tally_group.set('transform', transform)

        # Draw the rectangle outline
        rect = RectElement()
        rect.set('x', str(x_min))
        rect.set('y', str(y_min))
        rect.set('width', str(width))
        rect.set('height', str(height))
        rect.style = {
            'fill': 'none',
            'stroke': '#000000',
            'stroke-width': '0.5'
        }
        tally_group.append(rect)

        # Check which pricking points are connected to plaits
        control_path_id = control_path.get('id')
        connections = self.find_connections_to_control_path(control_path_id)

        # Determine which points are connected
        entry_connected = any(conn['pricking_index'] == 0 for conn in connections)
        exit_connected = any(conn['pricking_index'] == 1 for conn in connections)

        # Pricking points at entry and exit (top and bottom centers)
        point_radius = self.mm_to_units(0.8)

        # Top center (entry)
        top_circle = Circle()
        top_circle.set('cx', str(top_center[0]))
        top_circle.set('cy', str(top_center[1]))
        top_circle.set('r', str(point_radius))
        top_circle.style = {
            'fill': '#00FF00' if entry_connected else '#666666',
            'stroke': 'none'
        }
        tally_group.append(top_circle)

        # Bottom center (exit)
        bottom_circle = Circle()
        bottom_circle.set('cx', str(bottom_center[0]))
        bottom_circle.set('cy', str(bottom_center[1]))
        bottom_circle.set('r', str(point_radius))
        bottom_circle.style = {
            'fill': '#00FF00' if exit_connected else '#666666',
            'stroke': 'none'
        }
        tally_group.append(bottom_circle)

        # Connection markers at all 4 corners (red circles, outline only)
        marker_radius = self.mm_to_units(1.5)

        for corner in corners:
            marker = Circle()
            marker.set('cx', str(corner[0]))
            marker.set('cy', str(corner[1]))
            marker.set('r', str(marker_radius))
            marker.style = {
                'fill': 'none',
                'stroke': '#FF0000',
                'stroke-width': '0.5'
            }
            tally_group.append(marker)

        # Transform entry/exit points to world coordinates if we applied a transform
        if use_transform:
            transform = control_path.get('transform')
            if transform:
                from inkex import Transform
                trans = Transform(transform)
                top_center = list(trans.apply_to_point(top_center))
                bottom_center = list(trans.apply_to_point(bottom_center))
                corners = [list(trans.apply_to_point(c)) for c in corners]
            else:
                top_center = list(top_center)
                bottom_center = list(bottom_center)
                corners = [list(c) for c in corners]
        else:
            top_center = list(top_center)
            bottom_center = list(bottom_center)
            corners = [list(c) for c in corners]

        # Store metadata with transformed coordinates
        metadata = {
            'element_type': 'tally_rect',
            'pairs': 2,  # Tallies always have 2 pairs
            'width': width,
            'height': height,
            'entry_point': top_center,
            'exit_point': bottom_center,
            'connection_points': corners,
            'control_path_id': control_path.get('id')
        }

        tally_group.set('data-lace-metadata', json.dumps(metadata))

        # Add to layer
        layer.append(tally_group)

        # Hide or style the control path
        if self.options.show_control_path:
            control_path.style['opacity'] = '0.5'
            control_path.style['stroke'] = '#0000FF'
            control_path.style['stroke-dasharray'] = '5,5'
            control_path.style['fill'] = 'none'
        else:
            control_path.style['display'] = 'none'

    def create_tally_leaf(self, control_path, layer):
        """
        Create a leaf-shaped (lens/vesica piscis) tally element.

        Tally is a solid woven area with:
        - Lens outline (two symmetrical arcs meeting at points)
        - Pricking points only at entry and exit (top and bottom points)
        - Connection markers at top and bottom
        - Fixed at 2 thread pairs
        """
        from inkex import Ellipse as EllipseElement, Rectangle as RectElement, Transform

        # For ellipses and rectangles, get the actual dimensions and preserve transform
        # For other shapes (paths), use bounding box
        if isinstance(control_path, EllipseElement):
            # Get intrinsic ellipse dimensions (before transform)
            cx_attr = float(control_path.get('cx') or '0')
            cy_attr = float(control_path.get('cy') or '0')
            rx = float(control_path.get('rx') or '0')
            ry = float(control_path.get('ry') or '0')

            # Use the actual ellipse dimensions for the lens
            # The lens formula assumes height > width, so we need to ensure that
            # If the ellipse is wider than tall, we'll rotate it 90 degrees

            if rx > ry:
                # Horizontal ellipse - swap dimensions and add 90° rotation
                width = ry * 2
                height = rx * 2
                needs_rotation = True
            else:
                # Vertical ellipse - use as-is
                width = rx * 2
                height = ry * 2
                needs_rotation = False

            # Center in untransformed space
            cx = 0
            cy = 0

            # Lens shape is always oriented vertically (along Y-axis) in local space
            # Top and bottom points at ±(height/2)
            top_point = (0, -height / 2)
            bottom_point = (0, height / 2)

            # Build the complete transform
            # The composed_transform includes parent transforms and the ellipse's transform attribute,
            # but NOT the cx/cy translation. We need to:
            # 1. Rotate 90° if needed (for horizontal ellipses)
            # 2. Translate to (cx, cy)
            # 3. Apply the composed transform
            use_transform = True

            ellipse_composed_transform = control_path.composed_transform()

            # Build transform: composed @ translate(cx,cy) @ rotate(90 if needed)
            if needs_rotation:
                transform = ellipse_composed_transform @ Transform(f"translate({cx_attr},{cy_attr})") @ Transform("rotate(90)")
            else:
                transform = ellipse_composed_transform @ Transform(f"translate({cx_attr},{cy_attr})")

        elif isinstance(control_path, RectElement):
            # Get intrinsic rectangle dimensions (before transform)
            x_attr = float(control_path.get('x') or '0')
            y_attr = float(control_path.get('y') or '0')
            rect_width = float(control_path.get('width') or '0')
            rect_height = float(control_path.get('height') or '0')

            # Determine orientation - use longer dimension as height
            if rect_width > rect_height:
                # Horizontal rectangle - swap dimensions and add 90° rotation
                width = rect_height
                height = rect_width
                needs_rotation = True
            else:
                # Vertical rectangle - use as-is
                width = rect_width
                height = rect_height
                needs_rotation = False

            # Center in untransformed space (relative to rectangle origin)
            cx = rect_width / 2
            cy = rect_height / 2

            # Lens shape oriented vertically in local space
            # Top and bottom points relative to rectangle center
            if needs_rotation:
                # After 90° rotation, what was horizontal becomes vertical
                top_point = (0, -height / 2)
                bottom_point = (0, height / 2)
            else:
                top_point = (0, -height / 2)
                bottom_point = (0, height / 2)

            # Build the complete transform
            # composed_transform includes parent transforms and element's transform attribute
            use_transform = True
            rect_composed_transform = control_path.composed_transform()

            # Build transform: composed @ translate(x,y) @ translate_to_center @ rotate(90 if needed)
            if needs_rotation:
                transform = rect_composed_transform @ Transform(f"translate({x_attr},{y_attr})") @ Transform(f"translate({cx},{cy})") @ Transform("rotate(90)")
            else:
                transform = rect_composed_transform @ Transform(f"translate({x_attr},{y_attr})") @ Transform(f"translate({cx},{cy})")

        else:
            # Use bounding box for paths or other shapes
            bbox = control_path.bounding_box()

            if bbox is None:
                inkex.errormsg("Could not get bounding box of selected shape.")
                return

            # Extract dimensions from transformed bounding box
            x_min, x_max = bbox.left, bbox.right
            y_min, y_max = bbox.top, bbox.bottom
            width = x_max - x_min
            height = y_max - y_min

            # Center point
            cx = x_min + width / 2
            cy = y_min + height / 2

            # Top and bottom points (the sharp ends of the lens)
            top_point = (cx, y_min)
            bottom_point = (cx, y_max)

            use_transform = False

        # For a proper lens shape (vesica piscis) that fits exactly within W × H:
        # Two circles of radius R, centered at horizontal offset d from centerline
        #
        # Geometry:
        # - Circle centers at (cx ± d, cy)
        # - Both circles have radius R
        # - They intersect at top (cx, cy - H/2) and bottom (cx, cy + H/2)
        # - Maximum width is W at the center
        #
        # From the geometry:
        # d = (H² - W²) / (4W)
        # R = (H² + W²) / (4W)

        W = width
        H = height

        # Warn if the shape doesn't have a good aspect ratio for a lens
        # The longer dimension should be at least 1.5× the shorter dimension
        long_dim = max(W, H)
        short_dim = min(W, H)
        aspect_ratio = long_dim / short_dim if short_dim > 0 else 1.0
        if aspect_ratio < 1.5:
            inkex.errormsg(f"Warning: Leaf tally works best when one dimension is at least 1.5× the other. Current ratio: {aspect_ratio:.2f}")

        # Calculate offset and radius
        d = (H * H - W * W) / (4 * W)
        radius = (H * H + W * W) / (4 * W)

        # Create a group for this tally
        tally_group = Group()
        tally_group.set('id', self.svg.get_unique_id('tally_leaf_'))

        # Apply transform if using ellipse dimensions
        if use_transform:
            tally_group.set('transform', str(transform))

        # Draw the lens shape using two circular arcs
        # First arc: part of right circle, curves from top to bottom (rightward)
        # Second arc: part of left circle, curves from bottom to top (leftward)
        lens_path = PathElement()

        # For SVG arc: A rx ry x-axis-rotation large-arc-flag sweep-flag x y
        # Use small arcs (large-arc-flag = 0)

        path_data = (
            f"M {top_point[0]},{top_point[1]} "  # Start at top point
            f"A {radius},{radius} 0 0 1 {bottom_point[0]},{bottom_point[1]} "  # Arc to bottom (curving right)
            f"A {radius},{radius} 0 0 1 {top_point[0]},{top_point[1]} "  # Arc back to top (curving left)
            f"Z"  # Close path
        )

        lens_path.set('d', path_data)
        lens_path.style = {
            'fill': 'none',
            'stroke': '#000000',
            'stroke-width': '0.5'
        }
        tally_group.append(lens_path)

        # Entry and exit points are the top and bottom points of the lens
        top_center = top_point
        bottom_center = bottom_point

        # Check which pricking points are connected to plaits
        control_path_id = control_path.get('id')
        connections = self.find_connections_to_control_path(control_path_id)

        # Determine which points are connected
        entry_connected = any(conn['pricking_index'] == 0 for conn in connections)
        exit_connected = any(conn['pricking_index'] == 1 for conn in connections)

        # Pricking points at entry and exit
        point_radius = self.mm_to_units(0.8)

        # Top center (entry)
        top_circle = Circle()
        top_circle.set('cx', str(top_center[0]))
        top_circle.set('cy', str(top_center[1]))
        top_circle.set('r', str(point_radius))
        top_circle.style = {
            'fill': '#00FF00' if entry_connected else '#666666',
            'stroke': 'none'
        }
        tally_group.append(top_circle)

        # Bottom center (exit)
        bottom_circle = Circle()
        bottom_circle.set('cx', str(bottom_center[0]))
        bottom_circle.set('cy', str(bottom_center[1]))
        bottom_circle.set('r', str(point_radius))
        bottom_circle.style = {
            'fill': '#00FF00' if exit_connected else '#666666',
            'stroke': 'none'
        }
        tally_group.append(bottom_circle)

        # Connection markers at top and bottom (red circles, outline only)
        marker_radius = self.mm_to_units(1.5)

        for point in [top_center, bottom_center]:
            marker = Circle()
            marker.set('cx', str(point[0]))
            marker.set('cy', str(point[1]))
            marker.set('r', str(marker_radius))
            marker.style = {
                'fill': 'none',
                'stroke': '#FF0000',
                'stroke-width': '0.5'
            }
            tally_group.append(marker)

        # Transform points to world coordinates for metadata if needed
        if use_transform:
            top_center = list(transform.apply_to_point(top_center))
            bottom_center = list(transform.apply_to_point(bottom_center))
        else:
            top_center = list(top_center)
            bottom_center = list(bottom_center)

        # Store metadata with world coordinates
        metadata = {
            'element_type': 'tally_leaf',
            'pairs': 2,  # Tallies always have 2 pairs
            'width': width,
            'height': height,
            'entry_point': top_center,
            'exit_point': bottom_center,
            'connection_points': [top_center, bottom_center],
            'control_path_id': control_path.get('id')
        }

        tally_group.set('data-lace-metadata', json.dumps(metadata))

        # Add to layer
        layer.append(tally_group)

        # Hide or style the control path
        if self.options.show_control_path:
            control_path.style['opacity'] = '0.5'
            control_path.style['stroke'] = '#0000FF'
            control_path.style['stroke-dasharray'] = '5,5'
            control_path.style['fill'] = 'none'
        else:
            control_path.style['display'] = 'none'

    def create_plait(self, control_path, layer):
        """
        Create a plait element.

        Plait is a braided connector with:
        - Single STRAIGHT line from start to end (braiding creates tension, pulls straight)
        - NO pricking points along length (braided, not pinned)
        - Connection markers (red circles) at start and end points only
        - Optional picots at midpoint (two hollow circles + center pin dot)
        - Fixed at 2 thread pairs
        """
        # Get the path data
        path_data = control_path.get('d')
        path = Path(path_data)

        # Get start and end points from the control path
        subpaths = list(path.to_absolute().to_superpath())
        if not subpaths:
            inkex.errormsg("Invalid path for plait.")
            return

        subpath = subpaths[0]

        # Start point is the first point
        start_point = list(subpath[0][1])  # [handle_before, point, handle_after]

        # End point is the last point
        end_point = list(subpath[-1][1])

        # Snap distance in SVG units
        snap_dist = self.mm_to_units(self.options.snap_distance)

        # Try to snap start and end points to nearby pricking points
        start_connection = None
        end_connection = None

        # Check for snap at start point
        nearest_start = self.find_nearest_pricking_point(start_point, snap_dist)
        if nearest_start:
            # Snap to this pricking point
            start_point = list(nearest_start['point'])
            start_connection = nearest_start

        # Check for snap at end point
        nearest_end = self.find_nearest_pricking_point(end_point, snap_dist)
        if nearest_end:
            # Snap to this pricking point
            end_point = list(nearest_end['point'])
            end_connection = nearest_end

        # Create a group for this plait
        plait_group = Group()
        plait_group.set('id', self.svg.get_unique_id('plait_'))

        # Draw the plait as a STRAIGHT line from start to end
        # Plaits are braided, which creates tension and pulls them straight
        plait_path = PathElement()
        straight_path = f"M {start_point[0]},{start_point[1]} L {end_point[0]},{end_point[1]}"
        plait_path.set('d', straight_path)
        plait_path.style = {
            'fill': 'none',
            'stroke': '#000000',
            'stroke-width': '0.5'
        }
        plait_group.append(plait_path)

        # Connection markers at start and end (red circles, outline only)
        marker_radius = self.mm_to_units(1.5)

        for point in [start_point, end_point]:
            marker = Circle()
            marker.set('cx', str(point[0]))
            marker.set('cy', str(point[1]))
            marker.set('r', str(marker_radius))
            marker.style = {
                'fill': 'none',
                'stroke': '#FF0000',
                'stroke-width': '0.5'
            }
            plait_group.append(marker)

        # Add picots if requested
        if self.options.plait_picots:
            # Find midpoint of the STRAIGHT line
            mid_point = (
                (start_point[0] + end_point[0]) / 2,
                (start_point[1] + end_point[1]) / 2
            )

            # Calculate tangent along the straight line
            tangent = (
                end_point[0] - start_point[0],
                end_point[1] - start_point[1]
            )
            tangent_len = math.hypot(tangent[0], tangent[1])
            if tangent_len > 0:
                tangent = (tangent[0] / tangent_len, tangent[1] / tangent_len)
            else:
                tangent = (1, 0)

            # Calculate perpendicular to tangent for picot placement
            perp = (-tangent[1], tangent[0])

            # Picot parameters
            picot_offset = self.mm_to_units(1.5)  # Distance from centerline to picot circles
            picot_radius = self.mm_to_units(0.6)  # Radius of decorative circles
            pin_radius = self.mm_to_units(0.4)    # Radius of center pin dot

            # Left picot circle (hollow)
            left_picot_pos = (
                mid_point[0] + perp[0] * picot_offset,
                mid_point[1] + perp[1] * picot_offset
            )
            left_picot = Circle()
            left_picot.set('cx', str(left_picot_pos[0]))
            left_picot.set('cy', str(left_picot_pos[1]))
            left_picot.set('r', str(picot_radius))
            left_picot.style = {
                'fill': 'none',
                'stroke': '#000000',
                'stroke-width': '0.3'
            }
            plait_group.append(left_picot)

            # Right picot circle (hollow)
            right_picot_pos = (
                mid_point[0] - perp[0] * picot_offset,
                mid_point[1] - perp[1] * picot_offset
            )
            right_picot = Circle()
            right_picot.set('cx', str(right_picot_pos[0]))
            right_picot.set('cy', str(right_picot_pos[1]))
            right_picot.set('r', str(picot_radius))
            right_picot.style = {
                'fill': 'none',
                'stroke': '#000000',
                'stroke-width': '0.3'
            }
            plait_group.append(right_picot)

            # Center pin dot (filled)
            pin_dot = Circle()
            pin_dot.set('cx', str(mid_point[0]))
            pin_dot.set('cy', str(mid_point[1]))
            pin_dot.set('r', str(pin_radius))
            pin_dot.style = {
                'fill': '#666666',
                'stroke': 'none'
            }
            plait_group.append(pin_dot)

        # Store metadata including connection information
        metadata = {
            'element_type': 'plait',
            'pairs': 2,  # Plaits always have 2 pairs
            'has_picots': self.options.plait_picots,
            'start_point': start_point,
            'end_point': end_point,
            'connection_points': [start_point, end_point],
            'start_connection': start_connection,  # Connection info or None
            'end_connection': end_connection,      # Connection info or None
            'control_path_id': control_path.get('id')
        }

        plait_group.set('data-lace-metadata', json.dumps(metadata))

        # Add to layer
        layer.append(plait_group)

        # Hide or style the control path
        if self.options.show_control_path:
            control_path.style['opacity'] = '0.5'
            control_path.style['stroke'] = '#0000FF'
            control_path.style['stroke-dasharray'] = '5,5'
            control_path.style['fill'] = 'none'
        else:
            control_path.style['display'] = 'none'

    def create_leaf(self, control_path, layer):
        """
        Create a leaf element.

        Leaf is a decorative filled element with:
        - Closed outline path (leaf shape)
        - Pricking points around the perimeter at regular intervals
        - Central vein (midrib) running from base to tip
        - Entry/exit points at the base where the leaf attaches
        - Fixed at 4 thread pairs (2 pairs per side of vein)
        """
        # Get parameters
        spacing = self.calculate_spacing()

        # Create a group for this leaf
        leaf_group = Group()
        leaf_group.set('id', self.svg.get_unique_id('leaf_'))

        # Get the path data
        path_data = control_path.get('d')
        path = Path(path_data)

        # Build segments using superpath
        subpaths = list(path.to_absolute().to_superpath())

        if not subpaths:
            inkex.errormsg("Invalid path for leaf.")
            return

        # Process first subpath (should be a closed path)
        subpath = subpaths[0]

        # Build list of all points along the perimeter
        segments = []
        for i in range(len(subpath) - 1):
            p0 = subpath[i][1]      # Current endpoint
            p1 = subpath[i][2]      # First control point
            p2 = subpath[i+1][0]    # Second control point
            p3 = subpath[i+1][1]    # Next endpoint
            segments.append((p0, p1, p2, p3))

        if not segments:
            inkex.errormsg("Path is too short to generate leaf.")
            return

        # Sample the perimeter densely
        dense_points = []
        dense_tangents = []

        for seg in segments:
            # Sample each bezier segment with 50 points for accuracy
            for i in range(50):
                t = i / 50
                pt = self.cubic_bezier_point(seg[0], seg[1], seg[2], seg[3], t)
                tangent = self.cubic_bezier_tangent(seg[0], seg[1], seg[2], seg[3], t)
                dense_points.append(pt)
                dense_tangents.append(tangent)

        # Add final point to close the loop
        seg = segments[-1]
        pt = self.cubic_bezier_point(seg[0], seg[1], seg[2], seg[3], 1.0)
        tangent = self.cubic_bezier_tangent(seg[0], seg[1], seg[2], seg[3], 1.0)
        dense_points.append(pt)
        dense_tangents.append(tangent)

        # Calculate cumulative arc length
        cumulative_dist = [0]
        for i in range(1, len(dense_points)):
            dist = math.hypot(
                dense_points[i][0] - dense_points[i-1][0],
                dense_points[i][1] - dense_points[i-1][1]
            )
            cumulative_dist.append(cumulative_dist[-1] + dist)

        total_length = cumulative_dist[-1]
        if total_length == 0:
            inkex.errormsg("Path has zero length.")
            return

        # Resample at regular intervals for pricking points
        pricking_points = []
        target_dist = spacing / 2  # Start half a spacing in

        while target_dist < total_length - spacing * 0.1:
            # Find the bracketing dense points
            idx = 0
            for i in range(len(cumulative_dist) - 1):
                if cumulative_dist[i] <= target_dist <= cumulative_dist[i + 1]:
                    idx = i
                    break

            # Interpolate
            if idx < len(dense_points) - 1:
                segment_length = cumulative_dist[idx + 1] - cumulative_dist[idx]
                if segment_length > 0:
                    t_local = (target_dist - cumulative_dist[idx]) / segment_length
                else:
                    t_local = 0

                p1 = dense_points[idx]
                p2 = dense_points[idx + 1]

                point = (
                    p1[0] + t_local * (p2[0] - p1[0]),
                    p1[1] + t_local * (p2[1] - p1[1])
                )

                pricking_points.append({
                    'point': point,
                    'index': len(pricking_points),
                    'is_active': False
                })

            target_dist += spacing

        # Draw the leaf outline
        outline_path = PathElement()
        outline_path.set('d', path_data)
        outline_path.style = {
            'fill': 'none',
            'stroke': '#000000',
            'stroke-width': '0.5'
        }
        leaf_group.append(outline_path)

        # Draw the central vein (from first point to the point halfway around perimeter)
        if len(pricking_points) >= 2:
            base_point = pricking_points[0]['point']
            tip_index = len(pricking_points) // 2
            tip_point = pricking_points[tip_index]['point']

            vein_path = PathElement()
            vein_path.set('d', f"M {base_point[0]},{base_point[1]} L {tip_point[0]},{tip_point[1]}")
            vein_path.style = {
                'fill': 'none',
                'stroke': '#000000',
                'stroke-width': '0.3',
                'stroke-dasharray': '2,2'
            }
            leaf_group.append(vein_path)

        # Draw pricking points around the perimeter
        point_radius = self.mm_to_units(0.8)
        for pp in pricking_points:
            circle = Circle()
            circle.set('cx', str(pp['point'][0]))
            circle.set('cy', str(pp['point'][1]))
            circle.set('r', str(point_radius))
            circle.style = {
                'fill': '#666666',
                'stroke': 'none'
            }
            leaf_group.append(circle)

        # Add entry/exit markers at base (larger red circles)
        entry_exit_point = None
        if len(pricking_points) >= 1:
            marker_radius = self.mm_to_units(1.5)
            base_point = pricking_points[0]['point']
            entry_exit_point = list(base_point)

            marker = Circle()
            marker.set('cx', str(base_point[0]))
            marker.set('cy', str(base_point[1]))
            marker.set('r', str(marker_radius))
            marker.style = {
                'fill': 'none',
                'stroke': '#FF0000',
                'stroke-width': '0.5'
            }
            leaf_group.append(marker)

        # Store metadata
        metadata = {
            'element_type': 'leaf',
            'pairs': 4,  # Leaves typically use 4 pairs (2 per side of vein)
            'pricking_points': pricking_points,
            'entry_exit_point': entry_exit_point,
            'control_path_id': control_path.get('id')
        }

        leaf_group.set('data-lace-metadata', json.dumps(metadata))

        # Add to layer
        layer.append(leaf_group)

        # Hide or style the control path
        if self.options.show_control_path:
            control_path.style['opacity'] = '0.5'
            control_path.style['stroke'] = '#0000FF'
            control_path.style['stroke-dasharray'] = '5,5'
            control_path.style['fill'] = 'none'
        else:
            control_path.style['display'] = 'none'


if __name__ == '__main__':
    BedfordshireLace().run()
