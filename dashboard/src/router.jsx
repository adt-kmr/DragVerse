import { useSyncExternalStore } from "react";

/* Two routes, so two routes' worth of router. History API + popstate is the whole thing;
   swap in react-router the moment nested routes or params show up. */

const subscribe = (onChange) => {
  window.addEventListener("popstate", onChange);
  return () => window.removeEventListener("popstate", onChange);
};

export const usePath = () =>
  useSyncExternalStore(
    subscribe,
    () => window.location.pathname,
    () => "/"
  );

export function navigate(to) {
  if (to === window.location.pathname) return;
  window.history.pushState({}, "", to);
  window.dispatchEvent(new PopStateEvent("popstate"));
  window.scrollTo(0, 0);
}

/** An anchor that routes in-app but stays a real link — middle-click and
    open-in-new-tab keep working, which a <span onClick> would quietly break. */
export function Link({ to, children, ...rest }) {
  const onClick = (event) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
    event.preventDefault();
    navigate(to);
  };
  return (
    <a href={to} onClick={onClick} {...rest}>
      {children}
    </a>
  );
}
