import { withRouter } from 'next/router';
import NextLink from 'next/link';
import React, { Children, FC, forwardRef } from 'react';
const rootUrl = ['/index', '/'];

interface IProps {
	router: any
	children: any
	href: any
	activeClassName?: string
}

const ActiveLink: FC<IProps> = forwardRef((properties: IProps, ref) => {
	const { router, children, activeClassName, ...props } = properties;

	const child = Children.only(children);

	let className = child.props.className || '';
	const currentUrl = rootUrl.includes(router.pathname) ? '/' : router.pathname;
	if (currentUrl === props.href && activeClassName) {
		className = `${className} ${activeClassName}`.trim();
	}

	return (
		<NextLink {...props}>
			<a>{React.cloneElement(child, { ref, className, style: { width: '100%' } })}</a>
		</NextLink>
	);
});

export default withRouter(ActiveLink);