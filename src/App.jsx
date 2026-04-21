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
  const [activeTab, setActiveTab] = useState('environment');
  
  const trainingRef = useRef(null);
  const vizRef = useRef(null);

  const statsMemo = useMemo(() => {
    if (trainingStats.length === 0) return { successRate: 0, avgReward: 0 };
    const recent = trainingStats.slice(-10);
    const successRate = recent.filter(s => s.success).length / recent.length * 100;
    const avgReward = recent.reduce((sum, s) => sum + s.reward, 0) / recent.length;
    return { successRate: successRate.toFixed(1), avgReward: avgReward.toFixed(1) };
  }, [trainingStats]);

  const chartData = useMemo(() => {
    if (trainingStats.length === 0) return [];
    let recentSuccesses = [];
    let recentRewards = [];
    let cumulativeReward = 0;
    
    return trainingStats.map((stat) => {
      recentSuccesses.push(stat.success ? 1 : 0);
      recentRewards.push(stat.reward);
      cumulativeReward += stat.reward;
      
      if (recentSuccesses.length > 20) recentSuccesses.shift();
      if (recentRewards.length > 20) recentRewards.shift();
      
      const successRate = (recentSuccesses.reduce((a,b)=>a+b, 0) / recentSuccesses.length) * 100;
      const rollingAvgReward = recentRewards.reduce((a,b)=>a+b, 0) / recentRewards.length;
      
      return {
        ...stat,
        successRate: parseFloat(successRate.toFixed(1)),
        rollingAvgReward: parseFloat(rollingAvgReward.toFixed(1)),
        cumulativeReward
      };
    });
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

  // Render Q-Value Heatmap
  const renderHeatmap = () => {
    const cells = [];
    let minQ = -10, maxQ = 10; // Default range
    
    // Find min and max Q values to normalize colors
    if (Object.keys(agent.qTable).length > 0) {
      Object.values(agent.qTable).forEach(qVals => {
        const stateMax = Math.max(...qVals);
        const stateMin = Math.min(...qVals);
        if(stateMax > maxQ) maxQ = stateMax;
        if(stateMin < minQ) minQ = stateMin;
      });
    }

    for (let y = 0; y < env.height; y++) {
      for (let x = 0; x < env.width; x++) {
        const isObstacle = env.isObstacle(x, y);
        const isTarget = env.targetState.x === x && env.targetState.y === y;
        
        const stateKey = env.getStateKey({x, y});
        const qVals = agent.qTable[stateKey];
        let stateValue = 0;
        
        if (qVals && !isObstacle && !isTarget) {
          stateValue = Math.max(...qVals);
        } else if (isTarget) {
          stateValue = 100;
        } else if (isObstacle) {
          stateValue = -100;
        }

        let bgColor = 'var(--grid-bg)';
        if (isObstacle) {
          bgColor = 'rgba(239, 68, 68, 0.4)'; // Red
        } else if (isTarget) {
          bgColor = 'rgba(16, 185, 129, 0.5)'; // Green
        } else if (qVals) {
          if (stateValue > 0) {
            const intensity = Math.min(1, stateValue / (maxQ || 1));
            bgColor = `rgba(16, 185, 129, ${0.1 + intensity * 0.8})`;
          } else if (stateValue < 0) {
            const intensity = Math.min(1, stateValue / (minQ || -1));
            bgColor = `rgba(239, 68, 68, ${0.1 + intensity * 0.8})`;
          } else {
            bgColor = 'rgba(255, 255, 255, 0.05)';
          }
        }

        cells.push(
          <div key={`heat-${x}-${y}`} className={`grid-cell ${isObstacle ? 'obstacle' : ''} ${isTarget ? 'target' : ''}`} style={{ backgroundColor: bgColor }}>
            {isObstacle && <span className="heatmap-value">OBS</span>}
            {isTarget && <span className="heatmap-value">GOAL</span>}
            {!isObstacle && !isTarget && qVals && (
              <span className="heatmap-value">{stateValue.toFixed(1)}</span>
            )}
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
            <div className="tabs">
              <button 
                className={`tab-btn ${activeTab === 'environment' ? 'active' : ''}`}
                onClick={() => setActiveTab('environment')}
              >
                Simulation
              </button>
              <button 
                className={`tab-btn ${activeTab === 'heatmap' ? 'active' : ''}`}
                onClick={() => setActiveTab('heatmap')}
              >
                Q-Value Heatmap
              </button>
              <button 
                className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
                onClick={() => setActiveTab('analytics')}
              >
                Extended Analytics
              </button>
            </div>

            {activeTab === 'environment' && (
              <>
                <h2>Environment State</h2>
                <div className="grid-container">
                  <div className="grid-board">
                    {renderGrid()}
                  </div>
                </div>
              </>
            )}

            {activeTab === 'heatmap' && (
              <>
                <h2>Maximum Expected Reward (State Value)</h2>
                <div className="grid-container">
                  <div className="grid-board">
                    {renderHeatmap()}
                  </div>
                </div>
                <div style={{ textAlign: 'center', marginTop: '1rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  Green represents higher expected future rewards. Red indicates negative expected rewards (or obstacles).
                </div>
              </>
            )}

            {activeTab === 'analytics' && chartData.length > 0 && (
              <>
                <h2>Training Metrics (Rolling Averages)</h2>
                <div className="chart-container" style={{ marginTop: '0', height: '250px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="epoch" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                        itemStyle={{ color: '#10b981' }}
                      />
                      <Line type="monotone" dataKey="successRate" stroke="#10b981" name="Success Rate (%)" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                  <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.5rem', marginBottom: '1.5rem' }}>
                    Rolling Success Rate (20 Epochs)
                  </div>
                </div>

                <div className="chart-container" style={{ height: '250px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="epoch" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                        itemStyle={{ color: '#8b5cf6' }}
                      />
                      <Line type="monotone" dataKey="cumulativeReward" stroke="#8b5cf6" name="Cumulative Reward" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                  <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.5rem', marginBottom: '1.5rem' }}>
                    Cumulative Reward
                  </div>
                </div>
              </>
            )}

            {activeTab === 'environment' && trainingStats.length > 0 && (
              <>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="epoch" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                      itemStyle={{ color: '#38bdf8' }}
                    />
                    <Line type="monotone" dataKey="rollingAvgReward" name="Avg Reward (20 ep)" stroke="#38bdf8" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="reward" name="Reward" stroke="rgba(56, 189, 248, 0.3)" dot={false} strokeWidth={1} />
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.5rem' }}>
                  Reward Progression
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '2rem' }}>
                <div className="chart-container" style={{ marginTop: '0' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
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
                    <LineChart data={chartData}>
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
