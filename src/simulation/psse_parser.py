import re


class PSSEParser:
    """
    Minimal PSSE .raw parser (buses + lines)
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def parse(self):
        buses = []
        lines = []

        with open(self.filepath, 'r', errors='ignore') as f:
            data = f.readlines()

        mode = None

        for line in data:
            if "BEGIN BUS DATA" in line:
                mode = "bus"
                continue
            elif "END OF BUS DATA" in line:
                mode = None

            elif "BEGIN BRANCH DATA" in line:
                mode = "branch"
                continue
            elif "END OF BRANCH DATA" in line:
                mode = None

            if mode == "bus":
                parts = line.split(',')
                if len(parts) > 1:
                    try:
                        bus_id = int(parts[0])
                        buses.append(bus_id)
                    except:
                        pass

            elif mode == "branch":
                parts = line.split(',')
                if len(parts) > 2:
                    try:
                        from_bus = int(parts[0])
                        to_bus = int(parts[1])
                        lines.append((from_bus, to_bus))
                    except:
                        pass

        return buses, lines