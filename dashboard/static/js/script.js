/**
 * SUMO Traffic Dashboard JavaScript
 * Handles all dashboard interactions and real-time data updates
 */

// Dashboard state management
const DashboardState = {
    isConnected: false,
    currentConfig: null,
    lastUpdateTime: null,
    updateInterval: 500,     // OPTIMIZED: Reduced from 1000ms to 500ms
    statusInterval: 3000,    // OPTIMIZED: Increased from 2000ms to 3000ms
    dataCache: {},          // OPTIMIZATION: Add data caching
    lastDataUpdate: 0,
    manualMode: false,      // Add this
    statistics: {}          // Add this
};

/**
 * Categorize vehicles by direction based on their position
 * @param {Array} vehicles - Array of vehicle objects with x, y coordinates
 * @returns {Object} - Object with vehicles categorized by direction
 */
function categorizeVehiclesByDirection(vehicles) {
    const directions = { north: [], south: [], east: [], west: [] };
    
    vehicles.forEach(vehicle => {
        // OPTIMIZED: Simplified position-based logic (no road_id checking)
        const threshold = 25;
        
        if (vehicle.y > threshold) {
            directions.north.push(vehicle);
        } else if (vehicle.y < -threshold) {
            directions.south.push(vehicle);
        } else if (vehicle.x > threshold) {
            directions.east.push(vehicle);
        } else {
            directions.west.push(vehicle);
        }
    });
    
    return directions;
}

/**
 * Calculate congestion level based on vehicle count and average speed
 * @param {number} count - Number of vehicles
 * @param {number} avgSpeed - Average speed of vehicles
 * @returns {Object} - Congestion data with level, percentage, and className
 */
function calculateCongestion(count, avgSpeed) {
    let level = 'Low';
    let percentage = 0;
    let className = 'congestion-low';
    let textColor = 'text-green-600';
    
    if (count > 10 || avgSpeed < 5) {
        level = 'High';
        percentage = Math.min(100, (count * 10) + ((10 - avgSpeed) * 5));
        className = 'congestion-high';
        textColor = 'text-red-600';
    } else if (count > 5 || avgSpeed < 10) {
        level = 'Medium';
        percentage = Math.min(60, (count * 8) + ((15 - avgSpeed) * 3));
        className = 'congestion-medium';
        textColor = 'text-yellow-600';
    } else {
        percentage = Math.min(30, count * 5);
    }
    
    return { level, percentage, className, textColor };
}

/**
 * Update direction-specific data and UI elements
 * @param {string} direction - Direction name (north, south, east, west)
 * @param {Array} vehicles - Array of vehicles for this direction
 */
function updateDirectionData(direction, vehicles) {
    const count = vehicles.length;
    const avgSpeed = count > 0 ? vehicles.reduce((sum, v) => sum + v.speed, 0) / count : 0;
    const congestion = calculateCongestion(count, avgSpeed);
    
    // Update metrics with smooth animation
    animateValueChange(`${direction}-count`, count);
    animateValueChange(`${direction}-speed`, avgSpeed.toFixed(1));
    
    // Update congestion level and styling
    const congestionText = document.getElementById(`${direction}-congestion-text`);
    const congestionBar = document.getElementById(`${direction}-congestion-bar`);
    
    if (congestionText && congestionBar) {
        congestionText.textContent = congestion.level;
        congestionText.className = `font-bold ${congestion.textColor}`;
        
        congestionBar.style.width = congestion.percentage + '%';
        congestionBar.className = `h-full transition-all duration-500 ${congestion.className}`;
    }
    
    // Update vehicle list with enhanced styling
    updateVehicleList(direction, vehicles);
}

/**
 * Update vehicle list for a specific direction
 * @param {string} direction - Direction name
 * @param {Array} vehicles - Array of vehicles
 */
function updateVehicleList(direction, vehicles) {
    const vehicleList = document.getElementById(`${direction}-vehicles`);
    if (!vehicleList) return;
    
    if (vehicles.length === 0) {
        vehicleList.innerHTML = '<div class="text-gray-500 text-center py-4">No vehicles detected</div>';
        return;
    }
    
    vehicleList.innerHTML = vehicles.map(vehicle => `
        <div class="vehicle-item transform hover:scale-102 transition-all duration-200">
            <div class="flex justify-between items-center">
                <span class="font-semibold text-gray-800">${vehicle.id}</span>
                <span class="text-xs text-gray-500">${vehicle.speed} m/s</span>
            </div>
            <div class="text-xs text-gray-600 mt-1">
                Position: (${vehicle.x.toFixed(0)}, ${vehicle.y.toFixed(0)})
            </div>
        </div>
    `).join('');
}

/**
 * Animate value changes for smooth transitions
 * @param {string} elementId - ID of the element to update
 * @param {number|string} newValue - New value to display
 */
function animateValueChange(elementId, newValue) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const currentValue = parseFloat(element.textContent) || 0;
    const targetValue = parseFloat(newValue) || 0;
    
    // Add loading animation class
    element.classList.add('loading');
    
    setTimeout(() => {
        element.textContent = newValue;
        element.classList.remove('loading');
    }, 100);
}

/**
 * Fetch and update simulation data
 */
async function updateData() {
    const now = Date.now();
    
    // OPTIMIZATION: Throttle updates if called too frequently
    if (now - DashboardState.lastDataUpdate < 400) {
        return;
    }
    
    try {
        // Fetch main data
        const response = await fetch('/api/data');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        DashboardState.lastUpdateTime = new Date();
        DashboardState.lastDataUpdate = now;
        
        // OPTIMIZATION: Skip update if data hasn't changed
        if (JSON.stringify(data) === JSON.stringify(DashboardState.dataCache)) {
            return;
        }
        DashboardState.dataCache = data;
        
        // Update vehicle directions
        const directions = categorizeVehiclesByDirection(data.vehicles);
        Object.keys(directions).forEach(dir => {
            updateDirectionData(dir, directions[dir]);
        });
        
        updateTrafficLights(data);
        updateSummaryData(data);
        
        // Update statistics if available
        if (data.statistics) {
            updateStatistics(data.statistics);
        }
        
    } catch (error) {
        // OPTIMIZATION: Reduced error logging
        if (DashboardState.isConnected) { // Only log if we expect connection
            console.error('Data update failed');
        }
    }
}

function updateStatistics(stats) {
    // Update basic statistics
    animateValueChange('total-passed', stats.total_vehicles_passed || 0);
    animateValueChange('total-time', Math.floor((stats.total_simulation_time || 0) / 60));
    animateValueChange('congestion-events', (stats.congestion_events || []).length);
    animateValueChange('peak-vehicles', stats.peak_vehicle_count || 0);
    
    // Update congestion history
    const historyDiv = document.getElementById('congestion-history');
    if (historyDiv && stats.congestion_events) {
        const recentEvents = stats.congestion_events.slice(-5); // Last 5 events
        
        if (recentEvents.length === 0) {
            historyDiv.innerHTML = '<div class="text-gray-500 text-center">No congestion events yet</div>';
        } else {
            historyDiv.innerHTML = recentEvents.map(event => `
                <div class="bg-white rounded p-2 mb-2 border-l-4 border-orange-400">
                    <div class="flex justify-between text-sm">
                        <span>Time: ${Math.floor(event.time)}s</span>
                        <span>${event.vehicle_count} vehicles</span>
                    </div>
                    <div class="text-xs text-gray-600">
                        Avg Speed: ${event.avg_speed} m/s | Slow: ${event.slow_vehicles}
                    </div>
                </div>
            `).join('');
        }
    }
}

// Add this improved function that handles SUMO's lane-based traffic light states:

function updateTrafficLights(data) {
    const trafficLights = data.traffic_lights || {};
    
    // OPTIMIZED: Removed excessive console.log statements
    
    if (Object.keys(trafficLights).length === 0) {
        ['north', 'south', 'east', 'west'].forEach(direction => {
            const tlElement = document.getElementById(`${direction}-traffic-light`);
            if (tlElement) {
                tlElement.textContent = 'No Data';
                tlElement.className = 'ml-auto px-3 py-1 rounded-full text-sm font-medium bg-gray-400 text-white';
            }
        });
        return;
    }
    
    const mainTlId = Object.keys(trafficLights)[0];
    const tlData = trafficLights[mainTlId];
    const state = tlData.state || tlData;
    const isManual = tlData.manual_mode || false;
    
    // OPTIMIZED: Simplified direction mapping
    const directions = [
        { name: 'north', index: 0 },
        { name: 'east', index: 1 },
        { name: 'south', index: 2 },
        { name: 'west', index: 3 }
    ];
    
    directions.forEach(direction => {
        const tlElement = document.getElementById(`${direction.name}-traffic-light`);
        if (!tlElement) return;
        
        const charIndex = direction.index < state.length ? direction.index : 0;
        const tlState = state[charIndex] || 'r';
        
        let displayText = 'Red';
        let colorClass = 'bg-red-500 text-white';
        
        if (tlState === 'G' || tlState === 'g') {
            displayText = 'Green';
            colorClass = 'bg-green-500 text-white';
        } else if (tlState === 'Y' || tlState === 'y') {
            displayText = 'Yellow';
            colorClass = 'bg-yellow-500 text-black';
        }
        
        // Add manual mode indicator
        if (isManual) {
            displayText += ' (M)';
            colorClass += ' ring-2 ring-blue-400';
        }
        
        tlElement.textContent = displayText;
        tlElement.className = `ml-auto px-3 py-1 rounded-full text-sm font-medium ${colorClass}`;
    });
}

/**
 * Update summary dashboard data
 * @param {Object} data - Simulation data
 */
function updateSummaryData(data) {
    const totalVehicles = data.vehicles.length;
    const overallAvgSpeed = totalVehicles > 0 ? 
        data.vehicles.reduce((sum, v) => sum + v.speed, 0) / totalVehicles : 0;
    
    animateValueChange('total-vehicles', totalVehicles);
    animateValueChange('simulation-step', Math.floor(data.step));
    animateValueChange('avg-speed', overallAvgSpeed.toFixed(1));
    
    // Determine overall congestion status
    let overallStatus = 'Low';
    let statusColor = 'text-green-600';
    
    if (totalVehicles > 20 || overallAvgSpeed < 5) {
        overallStatus = 'High';
        statusColor = 'text-red-600';
    } else if (totalVehicles > 10 || overallAvgSpeed < 10) {
        overallStatus = 'Medium';
        statusColor = 'text-yellow-600';
    }
    
    const overallElement = document.getElementById('overall-congestion');
    if (overallElement) {
        overallElement.textContent = overallStatus;
        overallElement.className = `text-3xl font-bold ${statusColor}`;
    }
}

/**
 * Check connection status with SUMO
 */
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        const statusDiv = document.getElementById('status');
        
        if (!statusDiv) return;
        
        if (data.running) {
            DashboardState.isConnected = true;
            statusDiv.className = 'status-connected px-4 py-2 rounded-full font-semibold flex items-center gap-2';
            statusDiv.innerHTML = '<span class="w-3 h-3 rounded-full bg-current animate-pulse"></span> Status: Connected to SUMO';
        } else {
            DashboardState.isConnected = false;
            statusDiv.className = 'status-disconnected px-4 py-2 rounded-full font-semibold flex items-center gap-2';
            statusDiv.innerHTML = '<span class="w-3 h-3 rounded-full bg-current animate-pulse"></span> Status: Disconnected';
        }
        
    } catch (error) {
        console.error('Error checking status:', error);
        DashboardState.isConnected = false;
    }
}

/**
 * Start SUMO simulation
 */
async function startSimulation() {
    try {
        showNotification('Starting SUMO simulation...', 'info');
        
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        showNotification(data.message, 'success');
        
    } catch (error) {
        console.error('Error starting simulation:', error);
        showNotification('Failed to start simulation', 'error');
    }
}

/**
 * Stop SUMO simulation
 */
async function stopSimulation() {
    try {
        showNotification('Stopping SUMO simulation...', 'info');
        
        const response = await fetch('/api/stop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        showNotification(data.message, 'success');
        
    } catch (error) {
        console.error('Error stopping simulation:', error);
        showNotification('Failed to stop simulation', 'error');
    }
}

/**
 * Select a configuration file
 */
async function selectConfig() {
    const configSelect = document.getElementById('configSelect');
    if (!configSelect) return;
    
    const configName = configSelect.value;
    if (!configName) {
        showNotification('Please select a configuration file', 'warning');
        return;
    }
    
    try {
        showNotification(`Selecting configuration: ${configName}`, 'info');
        
        const response = await fetch('/api/select_config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ config_name: configName })
        });
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        DashboardState.currentConfig = configName;
        showNotification(data.message, 'success');
        
    } catch (error) {
        console.error('Error selecting config:', error);
        showNotification('Failed to select configuration', 'error');
    }
}

/**
 * Show notification to user
 * @param {string} message - Notification message
 * @param {string} type - Notification type (success, error, warning, info)
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 transform translate-x-full transition-all duration-300`;
    
    // Set color based on type
    const colors = {
        success: 'bg-green-500 text-white',
        error: 'bg-red-500 text-white',
        warning: 'bg-yellow-500 text-black',
        info: 'bg-blue-500 text-white'
    };
    
    notification.className += ` ${colors[type] || colors.info}`;
    notification.textContent = message;
    
    // Add to DOM
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.classList.remove('translate-x-full');
    }, 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.classList.add('translate-x-full');
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

/**
 * Initialize dashboard
 */
async function initializeDashboard() {
    console.log('Initializing SUMO Traffic Dashboard...');
    
    // Start periodic updates
    setInterval(updateData, DashboardState.updateInterval);
    setInterval(checkStatus, DashboardState.statusInterval);
    
    // Initial load
    checkStatus();
    updateData();
    initializeTrafficControl(); // Add this line
    
    console.log('Dashboard initialized successfully');
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', initializeDashboard);

// Handle page visibility changes to pause/resume updates
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Dashboard paused (tab not visible)');
    } else {
        console.log('Dashboard resumed (tab visible)');
        // Immediately update when tab becomes visible again
        checkStatus();
        updateData();
    }
});

// Debug function - run this in browser console while simulation is running
async function debugTrafficLights() {
    try {
        const response = await fetch('/api/debug/traffic_lights');
        const data = await response.json();
        console.log('=== TRAFFIC LIGHT DEBUG INFO ===');
        console.log(data);
        return data;
    } catch (error) {
        console.error('Debug failed:', error);
    }
}

// Run this in console:
// debugTrafficLights();

// Replace the toggleManualMode function:

async function toggleManualMode() {
    try {
        // First check if simulation is running
        const statusResponse = await fetch('/api/status');
        const statusData = await statusResponse.json();
        
        if (!statusData.running) {
            showNotification('Start SUMO simulation first before enabling manual mode', 'error');
            return;
        }
        
        const newMode = !DashboardState.manualMode;
        
        showNotification(`${newMode ? 'Enabling' : 'Disabling'} manual mode...`, 'info');
        
        const response = await fetch('/api/traffic_light/mode', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ manual_mode: newMode })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            showNotification(data.error, 'error');
            console.error('Manual mode error:', data.error);
            return;
        }
        
        DashboardState.manualMode = newMode;
        
        // Update UI
        const toggleButton = document.getElementById('manual-mode-toggle');
        const controlsDiv = document.getElementById('manual-controls');
        
        if (newMode) {
            toggleButton.textContent = 'ON';
            toggleButton.className = 'px-4 py-2 rounded-lg font-medium transition-all duration-200 bg-green-500 text-white';
            controlsDiv.style.display = 'grid';
            showNotification('Manual control enabled - All lights set to RED', 'success');
        } else {
            toggleButton.textContent = 'OFF';
            toggleButton.className = 'px-4 py-2 rounded-lg font-medium transition-all duration-200 bg-gray-400 text-white';
            controlsDiv.style.display = 'none';
            showNotification('Automatic control restored', 'info');
        }
        
    } catch (error) {
        console.error('Error toggling manual mode:', error);
        showNotification(`Failed to toggle manual mode: ${error.message}`, 'error');
    }
}

// Enhanced setTrafficLight function:
async function setTrafficLight(direction, state) {
    if (!DashboardState.manualMode) {
        showNotification('Enable manual mode first', 'warning');
        return;
    }
    
    try {
        showNotification(`Setting ${direction} to ${state}...`, 'info');
        
        const response = await fetch('/api/traffic_light/control', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                direction: direction, 
                state: state 
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            showNotification(data.error, 'error');
            console.error('Traffic control error:', data.error);
            return;
        }
        
        if (data.success) {
            showNotification(`${direction.toUpperCase()} set to ${state.toUpperCase()}`, 'success');
            console.log('Traffic light updated successfully:');
            console.log('- Current state:', data.current_state);
            console.log('- Verified state:', data.verified_state);
        }
        
    } catch (error) {
        console.error('Error controlling traffic light:', error);
        showNotification(`Failed to control traffic light: ${error.message}`, 'error');
    }
}

// Enhanced initialization for manual control status
async function initializeTrafficControl() {
    try {
        const response = await fetch('/api/traffic_light/status');
        if (response.ok) {
            const data = await response.json();
            DashboardState.manualMode = data.manual_mode;
            
            const toggleButton = document.getElementById('manual-mode-toggle');
            const controlsDiv = document.getElementById('manual-controls');
            
            if (DashboardState.manualMode) {
                toggleButton.textContent = 'ON';
                toggleButton.className = 'px-4 py-2 rounded-lg font-medium transition-all duration-200 bg-green-500 text-white';
                controlsDiv.style.display = 'grid';
            }
        }
    } catch (error) {
        console.error('Error initializing traffic control:', error);
    }
}