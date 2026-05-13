/*
 * NetworkNode.java — paste the "Additional class code" body into
 * AnyLogic's NetworkNode agent Properties → Additional class code.
 *
 * This agent represents one backbone PoP (router/node). It holds
 * its identity and population weight, and in a future extension
 * could also hold per-node state (queue length, current utilisation)
 * so the simulation can close the loop with the LSTM's predictions.
 */

// === ADDITIONAL CLASS CODE for NetworkNode ===

public int     nodeId     = 0;
public double  population = 1.0;

// total bytes received in the current slot (reset externally if needed)
public double  inboundThisSlot = 0.0;

public void recordInboundTraffic(int origin, double volume) {
    inboundThisSlot += volume;
}

public double drainInbound() {
    double v = inboundThisSlot;
    inboundThisSlot = 0.0;
    return v;
}

@Override
public String toString() {
    return "Node(" + nodeId + ", pop=" + String.format("%.2f", population) + ")";
}
