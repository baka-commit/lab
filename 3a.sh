
#!/bin/bash
clear

GREEN='\033[1;32m'
PURPLE='\033[1;35m'
YELLOW='\033[1;33m'
RESET='\033[0m'


show_prompt() {
    printf "${GREEN}%s@%s${RESET} ${PURPLE}UCRT64${RESET} ${YELLOW}%s${RESET}\n" \
        "$(whoami)" "$(hostname)" "$(pwd)"
}

show_command() {
    printf '$ %s\n' "$1"
}

show_prompt
show_command "node -v"
node -v
echo

show_prompt
show_command "npm -v"
npm -v
echo

show_prompt
show_command "npx create-react-app my-first-react-app"
npx create-react-app my-first-react-app 2>/dev/null
echo

cd my-first-react-app || exit 1

cat > src/App.js << 'EOF'
function App() {
  return (
    <div>
      <h1>Hello React</h1>
      <p>This is my first React application.</p>
    </div>
  );
}
export default App;
EOF

echo

show_prompt
show_command "npm start"
npm start 2>&1 | grep -v "DeprecationWarning"
