import React, { useState, useEffect, useRef, useMemo } from 'react';
import { DroneEnvironment, QAgent } from './rl/QAgent';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Play, FastForward, RotateCcw, BrainCircuit } from 'lucide-react';
import droneImg from './assets/drone.png';
import obstacleImg from './assets/obstacle.png';
import './index.css';

function App() {
  const [env] = useState(() => new DroneEnvironment());
  const [agent] = useState(() => new QAgent(env));
  
  const [dronePos, setDronePos] = useState({ ...env.startState });
  const [epoch, setEpoch] = useState(0);
  const [trainingStats, setTrainingStats] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isVisualizing, setIsVisualizing] = useState(false);
  const [epsilon, setEpsilon] = useState(agent.epsilon);
  
  const trainingRef = useRef(null);
  const vizRef = useRef(null);

  const statsMemo = useMemo(() => {
    if (trainingStats.length === 0) return { successRate: 0, avgReward: 0 };
    const recent = trainingStats.slice(-10);
    const successRate = recent.filter(s => s.success).length / recent.length * 100;
    const avgReward = recent.reduce((sum, s) => sum + s.reward, 0) / recent.length;
    return { successRate: successRate.toFixed(1), avgReward: avgReward.toFixed(1) };
  }, [trainingStats]);

  // Fast training
  const trainEpochs = (numEpochs) => {
    if (isTraining || isVisualizing) return;
    setIsTraining(true);
    
    let currentEpoch = epoch;
    const stats = [...trainingStats];
    
    const trainBatch = () => {
      const batchSize = Math.min(10, numEpochs - (currentEpoch - epoch));
      for (let i = 0; i < batchSize; i++) {
        const result = agent.trainEpoch();
        currentEpoch++;
        stats.push({
          epoch: currentEpoch,
          reward: result.totalReward,
          steps: result.steps,
          success: result.success,
          epsilon: agent.epsilon
        });
      }
      
      setEpoch(currentEpoch);
      setTrainingStats([...stats]);
      setEpsilon(agent.epsilon);
      
      if (currentEpoch - epoch < numEpochs) {
        trainingRef.current = requestAnimationFrame(trainBatch);
      } else {
        setIsTraining(false);
      }
    };
    
    trainingRef.current = requestAnimationFrame(trainBatch);
  };

  const stopTraining = () => {
    if (trainingRef.current) cancelAnimationFrame(trainingRef.current);
    if (vizRef.current) clearTimeout(vizRef.current);
    setIsTraining(false);
    setIsVisualizing(false);
  };

  const resetAll = () => {
    stopTraining();
    const newEnv = new DroneEnvironment();
    const newAgent = new QAgent(newEnv);
    // Mutate state directly instead of setting new instances to avoid complex re-renders 
    // for this simple simulation, or we can just reload
    window.location.reload();
  };

  // Visualize one episode using learned policy (epsilon = 0)
  const visualizeEpisode = () => {
    if (isTraining || isVisualizing) return;
    setIsVisualizing(true);
    
    // Save original epsilon and set to 0 to only exploit
    const originalEpsilon = agent.epsilon;
    agent.epsilon = 0;
    
    let state = env.reset();
    setDronePos({ ...env.currentState });
    
    let steps = 0;
    
    const stepViz = () => {
      if (env.isDone || steps > 50) {
        agent.epsilon = originalEpsilon;
        setIsVisualizing(false);
        return;
      }
      
      const action = agent.chooseAction(state);
      const { state: nextState, reward, done } = env.step(action);
      
      // We still update Q-values even during visualization to keep learning
      agent.learn(state, action, reward, nextState);
      
      state = nextState;
      setDronePos({ ...env.currentState });
      steps++;
      
      vizRef.current = setTimeout(stepViz, 200); // 200ms delay for visualization
    };
    
    vizRef.current = setTimeout(stepViz, 200);
  };

  // Render grid
  const renderGrid = () => {
    const cells = [];
    for (let y = 0; y < env.height; y++) {
      for (let x = 0; x < env.width; x++) {
        const isObstacle = env.isObstacle(x, y);
        const isTarget = env.targetState.x === x && env.targetState.y === y;
        const isDrone = dronePos.x === x && dronePos.y === y;
        
        // Visualize best action direction if we have some knowledge
        const stateKey = env.getStateKey({x, y});
        const qVals = agent.qTable[stateKey];
        let bestAction = -1;
        if (qVals && !isObstacle && !isTarget) {
          const maxQ = Math.max(...qVals);
          if (maxQ !== 0) {
            bestAction = qVals.indexOf(maxQ);
          }
        }

        cells.push(
          <div key={`${x}-${y}`} className={`grid-cell ${isObstacle ? 'obstacle' : ''} ${isTarget ? 'target' : ''}`}>
            {!isObstacle && !isTarget && !isDrone && qVals && (
              <div className="q-arrows">
                <div className={`arrow arrow-up`} style={{ borderBottomColor: bestAction === 0 ? 'var(--success)' : (qVals[0] > 0 ? 'rgba(255,255,255,0.3)' : 'transparent'), opacity: bestAction===0?1:0.3 }}></div>
                <div className={`arrow arrow-right`} style={{ borderLeftColor: bestAction === 1 ? 'var(--success)' : (qVals[1] > 0 ? 'rgba(255,255,255,0.3)' : 'transparent'), opacity: bestAction===1?1:0.3 }}></div>
                <div className={`arrow arrow-down`} style={{ borderTopColor: bestAction === 2 ? 'var(--success)' : (qVals[2] > 0 ? 'rgba(255,255,255,0.3)' : 'transparent'), opacity: bestAction===2?1:0.3 }}></div>
                <div className={`arrow arrow-left`} style={{ borderRightColor: bestAction === 3 ? 'var(--success)' : (qVals[3] > 0 ? 'rgba(255,255,255,0.3)' : 'transparent'), opacity: bestAction===3?1:0.3 }}></div>
              </div>
            )}
            
            {isDrone && <img src={droneImg} alt="drone" className="drone-image" />}
            {isTarget && !isDrone && <div className="target-icon">🎯</div>}
            {isObstacle && !isDrone && !isTarget && <img src={obstacleImg} alt="obstacle" className="obstacle-image" />}
          </div>
        );
      }
    }
    return cells;
  };

  return (
    <div className="app-container">
      <header>
        <h1>Autonomous Drone Path Planning</h1>
        <p className="subtitle">Simulated Environment Using Reinforcement Learning</p>
      </header>
      
      <div className="dashboard">
        <div className="main-content">
          <div className="panel">
            <h2>Simulation Environment</h2>
            <div className="grid-container">
              <div className="grid-board">
                {renderGrid()}
              </div>
            </div>
            
            {trainingStats.length > 0 && (
              <>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingStats}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="epoch" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                      itemStyle={{ color: '#38bdf8' }}
                    />
                    <Line type="monotone" dataKey="reward" stroke="#38bdf8" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.5rem' }}>
                  Reward Progression
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '2rem' }}>
                <div className="chart-container" style={{ marginTop: '0' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingStats}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="epoch" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                        itemStyle={{ color: '#10b981' }}
                      />
                      <Line type="monotone" dataKey="steps" stroke="#10b981" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                  <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.5rem' }}>
                    Steps per Episode
                  </div>
                </div>
                
                <div className="chart-container" style={{ marginTop: '0' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingStats}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="epoch" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                        itemStyle={{ color: '#f59e0b' }}
                      />
                      <Line type="monotone" dataKey="epsilon" stroke="#f59e0b" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                  <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.5rem' }}>
                    Exploration Rate (ε) Decay
                  </div>
                </div>
              </div>
              </>
            )}
          </div>
        </div>
        
        <div className="sidebar">
          <div className="panel">
            <h2>Agent Status</h2>
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-value">{epoch}</div>
                <div className="stat-label">Epochs</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{epsilon.toFixed(3)}</div>
                <div className="stat-label">Exploration (ε)</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{statsMemo.successRate}%</div>
                <div className="stat-label">Recent Success</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{statsMemo.avgReward}</div>
                <div className="stat-label">Avg Reward</div>
              </div>
            </div>
            
            <h2 style={{marginTop: '2rem'}}>Controls</h2>
            <div className="controls">
              <button 
                onClick={() => trainEpochs(10)} 
                disabled={isTraining || isVisualizing}
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
              >
                <BrainCircuit size={18} /> Train 10 Epochs
              </button>
              <button 
                onClick={() => trainEpochs(100)} 
                disabled={isTraining || isVisualizing}
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
              >
                <FastForward size={18} /> Train 100 Epochs
              </button>
              <button 
                className="secondary"
                onClick={visualizeEpisode} 
                disabled={isTraining || isVisualizing}
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
              >
                <Play size={18} /> Visualize Path
              </button>
              <button 
                className="secondary"
                onClick={resetAll} 
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', marginTop: '1rem', borderColor: 'var(--grid-border)', color: 'var(--text-muted)' }}
              >
                <RotateCcw size={18} /> Reset Environment
              </button>
            </div>
            
            <div style={{ marginTop: '2rem', fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
              <p><strong>Parameters:</strong></p>
              <ul style={{ paddingLeft: '1.2rem', marginTop: '0.5rem' }}>
                <li>Learning Rate (α): 0.1</li>
                <li>Discount Factor (γ): 0.9</li>
                <li>Min Exploration: 0.01</li>
              </ul>
              <p style={{ marginTop: '1rem' }}>The drone learns to navigate to the target (🎯) while avoiding red obstacles and minimizing energy consumption (steps).</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
