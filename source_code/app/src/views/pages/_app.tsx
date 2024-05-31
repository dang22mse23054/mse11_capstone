import React, { Fragment } from 'react';
import Head from 'next/head';
import '../public/styles/globals.css';
import type { AppProps /*, AppContext */ } from 'next/app';

function MyApp({ Component, pageProps }: AppProps) {
	return (
		<Fragment>
			<Head>
				<title>{process.env.WEB_TITLE ? process.env.WEB_TITLE : 'Ads Tracker'}</title>
			</Head>
			<Component {...pageProps} />
		</Fragment>
	);
}

export default MyApp;