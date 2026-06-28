import { useEffect } from "react";
import { useLocation } from "react-router-dom";

/**
 * Resets window scroll to the top whenever the route pathname
 * changes. React Router does NOT do this by default — without it,
 * navigating from a page scrolled to 60 % into a new page leaves
 * the new page scrolled to 60 % of its content, which reads as a
 * bug.
 *
 * Mount inside <BrowserRouter> exactly once. Renders nothing.
 */
export function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    // 'instant' so it doesn't visually animate during navigation.
    window.scrollTo({ top: 0, left: 0, behavior: "instant" });
  }, [pathname]);
  return null;
}
