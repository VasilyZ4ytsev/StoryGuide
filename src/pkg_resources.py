from importlib.metadata import entry_points


class WorkingSet:
    """Minimal shim for pymorphy2 on modern setuptools/Python."""

    def iter_entry_points(self, group, name=None):
        points = entry_points()
        if hasattr(points, "select"):
            selected = points.select(group=group)
        else:
            selected = points.get(group, [])

        for point in selected:
            if name is None or point.name == name:
                yield point
