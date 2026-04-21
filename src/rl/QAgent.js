export class DroneEnvironment {
  constructor(width = 10, height = 10) {
    this.width = width;
    this.height = height;
    this.startState = { x: 0, y: 0 };
    this.targetState = { x: 9, y: 9 };
    this.obstacles = [
      { x: 3, y: 3 }, { x: 3, y: 4 }, { x: 3, y: 5 },
      { x: 6, y: 6 }, { x: 7, y: 6 }, { x: 8, y: 6 },
      { x: 5, y: 1 }, { x: 5, y: 2 }, { x: 1, y: 8 },
      { x: 2, y: 8 }
    ];
    this.reset();
  }

  reset() {
    this.currentState = { ...this.startState };
    this.isDone = false;
    this.steps = 0;
    return this.getStateKey(this.currentState);
  }

  getStateKey(state) {
    return `${state.x},${state.y}`;
  }

  isObstacle(x, y) {
    return this.obstacles.some(o => o.x === x && o.y === y);
  }

  // Actions: 0: Up, 1: Right, 2: Down, 3: Left
  step(action) {
    if (this.isDone) return { state: this.getStateKey(this.currentState), reward: 0, done: true };

    this.steps++;
    let { x, y } = this.currentState;

    if (action === 0) y -= 1; // Up
    else if (action === 1) x += 1; // Right
    else if (action === 2) y += 1; // Down
    else if (action === 3) x -= 1; // Left

    // Check boundaries
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
      this.isDone = true;
      return { state: this.getStateKey(this.currentState), reward: -50, done: true, hitType: 'boundary' };
    }

    // Check obstacles
    if (this.isObstacle(x, y)) {
      this.isDone = true;
      this.currentState = { x, y }; // Move into obstacle for visualization
      return { state: this.getStateKey({x,y}), reward: -100, done: true, hitType: 'obstacle' };
    }

    this.currentState = { x, y };

    // Check target
    if (x === this.targetState.x && y === this.targetState.y) {
      this.isDone = true;
      return { state: this.getStateKey(this.currentState), reward: 100, done: true, hitType: 'target' };
    }

    // Regular step (energy penalty & distance)
    // Small shaped reward to encourage moving towards target
    const distToTarget = Math.abs(x - this.targetState.x) + Math.abs(y - this.targetState.y);
    const prevDist = Math.abs(this.currentState.x - this.targetState.x) + Math.abs(this.currentState.y - this.targetState.y);
    // actually prevDist is same right now, we need to save previous.
    // Let's just use a simple step penalty
    return { state: this.getStateKey(this.currentState), reward: -1, done: false, hitType: 'step' };
  }
}

export class QAgent {
  constructor(env, alpha = 0.1, gamma = 0.9, epsilon = 1.0, epsilonDecay = 0.995, minEpsilon = 0.01) {
    this.env = env;
    this.qTable = {};
    this.alpha = alpha; // Learning rate
    this.gamma = gamma; // Discount factor
    this.epsilon = epsilon; // Exploration rate
    this.epsilonDecay = epsilonDecay;
    this.minEpsilon = minEpsilon;
    this.actions = [0, 1, 2, 3];
  }

  getQValues(state) {
    if (!this.qTable[state]) {
      this.qTable[state] = [0, 0, 0, 0];
    }
    return this.qTable[state];
  }

  chooseAction(state) {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actions.length);
    } else {
      const qValues = this.getQValues(state);
      const maxQ = Math.max(...qValues);
      const bestActions = this.actions.filter(a => qValues[a] === maxQ);
      return bestActions[Math.floor(Math.random() * bestActions.length)];
    }
  }

  learn(state, action, reward, nextState) {
    const qValues = this.getQValues(state);
    const nextQValues = this.getQValues(nextState);
    const maxNextQ = Math.max(...nextQValues);

    qValues[action] = qValues[action] + this.alpha * (reward + this.gamma * maxNextQ - qValues[action]);
    this.qTable[state] = qValues;
  }

  decayEpsilon() {
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
  }

  // Train for one epoch (episode) without visualizing
  trainEpoch() {
    let state = this.env.reset();
    let totalReward = 0;
    let done = false;
    let steps = 0;

    while (!done && steps < 100) { // Max 100 steps per episode
      const action = this.chooseAction(state);
      const { state: nextState, reward, done: isDone, hitType } = this.env.step(action);
      
      this.learn(state, action, reward, nextState);
      
      state = nextState;
      totalReward += reward;
      done = isDone;
      steps++;
    }

    this.decayEpsilon();
    return {
      totalReward,
      steps,
      success: this.env.currentState.x === this.env.targetState.x && this.env.currentState.y === this.env.targetState.y
    };
  }
}
