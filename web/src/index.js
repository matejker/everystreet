import React from "react";
import ReactDOM from "react-dom";
import { HashRouter as Router, Switch, Route } from "react-router-dom";
import { Home } from "./pages/Home";
import { About } from "./pages/About";
import { ShowRoute } from "./pages/ShowRoute";
import { PrivacyPolicy } from "./pages/PrivacyPolicy"
import { Changelog } from "./pages/Changelog"

export const Index = () => {

  return (
    <Router>
        <Switch>
          <Route path="/about" component={About} />
          {/* <Route path={"/route/:routeId"} component={ShowRoute} /> */}
          <Route path={"/privacy-policy"} component={PrivacyPolicy} />
          <Route path={"/changelog"} component={Changelog} />
          <Route path="/" component={Home} />
        </Switch>
    </Router>
  );
}

const rootElement = document.getElementById("root");
ReactDOM.render(
  <React.StrictMode>
    <Index />
  </React.StrictMode>,
  rootElement
);
