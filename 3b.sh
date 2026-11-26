
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
show_command "npx create-react-app react-router-app"
npx create-react-app react-router-app 2>/dev/null
echo

cd react-router-app || exit 1

show_prompt
show_command "npm install react-router-dom"
npm install react-router-dom
echo

cat > src/Home.js << 'EOF'
function Home() {
  return (
    <div>
      <h2>Home Page</h2>
      <p>Welcome to the React Router application.</p>
    </div>
  );
}
export default Home;
EOF
echo

cat > src/About.js << 'EOF'
function About() {
  return (
    <div>
      <h2>About Page</h2>
      <p>This page explains the application.</p>
    </div>
  );
}
export default About;
EOF
echo

cat > src/Contact.js << 'EOF'
function Contact() {
  return (
    <div>
      <h2>Contact Page</h2>
      <p>Email: admin@example.com</p>
    </div>
  );
}
export default Contact;
EOF
echo

cat > src/App.js << 'EOF'
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Home from "./Home";
import About from "./About";
import Contact from "./Contact";

function App() {
  return (
    <BrowserRouter>
      <div>
        <h1>My React Website</h1>
        <nav>
          <Link to="/">Home</Link> |{" "}
          <Link to="/about">About</Link> |{" "}
          <Link to="/contact">Contact</Link>
        </nav>
        <hr/>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
export default App;
EOF
echo

show_prompt
show_command "npm start"
npm start 2>&1 | grep -v "DeprecationWarning"
