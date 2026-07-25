import React from "react";

export const About = () => {
  return (
      <>
        <h3>How does it work?</h3>
        <p>All you have to do is draw a <a href="https://en.wikipedia.org/wiki/Polygon">polygon</a> on
        the map. You can either draw your own or search for a neighbourhood, and you can adjust, remove, or create
        a new polygon at any time. Once your selected area is ready, hit the <em>Generate route</em> button
        and your optimal route will be computed right in your browser.</p>

        <p>When your route is ready, you will see a map with the blueprint of your street network.
        Then hit <em>Run the route</em> to animate it.</p>

        <h3>Is it really the most optimal route?</h3>
        <p>The short answer is yes &mdash; it <em>should</em> be optimal, because it visits every street at least once
        and is the shortest such route possible. For a sketch of the
        proof, see the <a href='everystreet_algorithm.pdf'>theoretical summary</a> or the <a href="https://github.com/matejker/everystreet">GitHub repo</a>. For the full
        proof, see <a href="https://doi.org/10.1007/BF01580113">Matching, Euler tours and the Chinese postman</a>, written
        by Edmonds and Johnson in 1973.</p>

        <h3>Why can't I generate an area larger than 3&nbsp;km<sup>2</sup>?</h3>
        <p>The #everystreet algorithm is fairly computationally heavy, and it now runs entirely in your browser, so the
        area is limited to 3&nbsp;km<sup>2</sup>. You can always split your town into smaller chunks and generate them
        piece by piece. If you really want to generate a route for a larger area, you can run the algorithm on your
        own machine &mdash; see the <a href="https://github.com/matejker/everystreet">GitHub repo</a>.</p>

        <h3>Why does the algorithm ignore some streets within the selected area?</h3>
        <p>Good question. The term <em>street</em> is ambiguous, and in general it is hard to define what counts as a
        street and what does not. For simplicity, we consider all roads accessible by car that start and end
        within the selected area. To determine these roads, we use <em>OpenStreetMap</em>'s <em>drive</em> layer.</p>
      </>
  );
}
